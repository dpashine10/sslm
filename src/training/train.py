"""Train the selective-SSM language model.

The default command is intentionally explicit about data and token budgets:

``python -m src.training.train --config configs/tiny.yaml --dry-run``
``python -m src.training.train --config configs/base_50m.yaml``
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:  # Repository-root module invocation.
    from src.data.dataset import RandomWindowDataset
    from src.models import CausalLMOutput
except ImportError:  # pragma: no cover - installed package invocation
    from data.dataset import RandomWindowDataset
    from models import CausalLMOutput
from .checkpointing import (
    atomic_torch_save,
    load_checkpoint,
    load_latest_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
    save_inference_export,
)
from .utils import (
    autocast_context,
    build_model,
    choose_device,
    environment_report,
    infer_vocab_size,
    load_yaml,
    make_scaler,
    model_config_dict,
    seed_everything,
    select_precision,
    write_json,
)


def _nested(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train = config.get("training", {})
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(_nested(train, "learning_rate", 3e-4)),
        betas=tuple(_nested(train, "betas", [0.9, 0.95])),
        eps=float(_nested(train, "adam_eps", 1e-8)),
        weight_decay=float(_nested(train, "weight_decay", 0.1)),
    )


def _print_environment(device: torch.device, precision: str, model: torch.nn.Module) -> None:
    report = environment_report(device, precision)
    print(json.dumps({**report, "parameters": sum(p.numel() for p in model.parameters())}, indent=2))


def _batch_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    # The model shifts internally: logits[:, :-1] predicts input_ids[:, 1:].
    with autocast_context(device, precision):
        output = model(input_ids, targets=input_ids)
    if not isinstance(output, CausalLMOutput) or output.loss is None:
        raise RuntimeError("model did not return a causal loss")
    return output.loss


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    max_batches: int | None = None,
) -> tuple[float, int, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    tokens = 0
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        loss = _batch_loss(model, batch, device, precision)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss at batch {index}: {loss.item()}")
        losses.append(float(loss.detach().cpu()))
        tokens += int(batch["input_ids"].numel())
    if was_training:
        model.train()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return (sum(losses) / max(1, len(losses)), tokens, tokens / elapsed)


def _tokenizer_encode(path: str | Path, prompt: str, max_length: int) -> torch.Tensor | None:
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    if not tokenizer_path.is_file():
        return None
    ids = Tokenizer.from_file(str(tokenizer_path)).encode(prompt).ids
    if not ids:
        return None
    return torch.tensor([ids[-max_length:]], dtype=torch.long)


@torch.no_grad()
def print_samples(model: torch.nn.Module, config: dict[str, Any], device: torch.device) -> None:
    generation = config.get("generation", {})
    tokenizer_path = generation.get("tokenizer") or config.get("tokenizer")
    if not tokenizer_path:
        return
    prompts = generation.get(
        "prompts",
        [
            "The history of computers",
            "Once upon a time",
            "Explain why the sky appears blue.",
            "Write a short Python function that adds two numbers.",
            "User: Hello! Assistant:",
        ],
    )
    if not isinstance(prompts, list):
        return
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return
    tokenizer_path_object = Path(tokenizer_path)
    if tokenizer_path_object.is_dir():
        tokenizer_path_object /= "tokenizer.json"
    if not tokenizer_path_object.is_file():
        return
    tokenizer_object = Tokenizer.from_file(str(tokenizer_path_object))
    model_was_training = model.training
    model.eval()
    for prompt in prompts:
        encoded = _tokenizer_encode(tokenizer_path, str(prompt), model.config.max_seq_len)
        if encoded is None:
            continue
        result = model.generate(
            encoded.to(device),
            int(_nested(generation, "max_new_tokens", 64)),
            temperature=float(_nested(generation, "temperature", 0.8)),
            top_k=int(_nested(generation, "top_k", 50)),
            top_p=float(_nested(generation, "top_p", 0.95)),
            do_sample=bool(_nested(generation, "do_sample", True)),
        )
        text = tokenizer_object.decode(result[0].tolist(), skip_special_tokens=False)
        print(f"\n[SAMPLE] {prompt}\n{text}")
    if model_was_training:
        model.train()


def _save_best(path: Path, output_dir: Path) -> None:
    payload = load_checkpoint(path, map_location="cpu")
    payload["protected"] = True
    atomic_torch_save(payload, output_dir / "best.pt")


def dry_run(config: dict[str, Any], device: torch.device, precision: str) -> dict[str, Any]:
    """Run one complete model/optimizer/generation check without data files."""

    seed_everything(int(_nested(config, "seed", 1234)))
    model = build_model(config, vocab_size=int(_nested(config, "dry_run_vocab_size", 256)))
    model.to(device)
    optimizer = _optimizer(model, config)
    scaler = make_scaler(device, precision)
    batch_size = int(_nested(config.get("training", {}), "batch_size", 2))
    sequence_length = min(int(model.config.max_seq_len), int(_nested(config.get("training", {}), "block_size", 16)))
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, max(2, sequence_length)), device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = _batch_loss(model, {"input_ids": input_ids}, device, precision)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"dry-run produced non-finite loss: {loss.item()}")
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    generated = model.generate(input_ids[:, : min(4, input_ids.size(1))], 4)
    result = {
        "loss": float(loss.detach().cpu()),
        "parameters": model.num_parameters(),
        "input_shape": list(input_ids.shape),
        "generated_shape": list(generated.shape),
        "device": str(device),
        "precision": precision,
    }
    print(json.dumps(result, indent=2))
    return result


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(_nested(config, "seed", 1234)))
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    device = choose_device(str(_nested(config, "device", "auto")))
    precision = select_precision(str(_nested(config, "precision", "auto")), device)
    token_manifest = data_cfg.get("token_manifest") or config.get("token_manifest")
    if not token_manifest:
        raise ValueError("data.token_manifest is required for training")
    vocab_size = infer_vocab_size(token_manifest)
    model = build_model(config, vocab_size=vocab_size).to(device)
    _print_environment(device, precision, model)

    block_size = int(_nested(train_cfg, "block_size", model.config.max_seq_len))
    if block_size > model.config.max_seq_len:
        raise ValueError("training.block_size cannot exceed model.max_seq_len")
    batch_size = int(_nested(train_cfg, "batch_size", 8))
    grad_accum = int(_nested(train_cfg, "gradient_accumulation_steps", 1))
    if batch_size < 1 or grad_accum < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    train_set = RandomWindowDataset(token_manifest, "train", block_size)
    validation_set = None
    try:
        validation_set = RandomWindowDataset(token_manifest, "validation", block_size)
    except (FileNotFoundError, ValueError):
        print("No validation shards found; validation will be skipped")
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = (
        DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
        if validation_set is not None
        else None
    )
    if len(loader) == 0:
        raise ValueError("Training data has no complete windows for the configured block_size")

    tokens_per_step = batch_size * block_size * grad_accum
    max_tokens = train_cfg.get("max_tokens")
    max_steps = train_cfg.get("max_steps")
    if max_steps is None:
        if max_tokens is None:
            raise ValueError("set training.max_steps or training.max_tokens")
        max_steps = max(1, math.ceil(int(max_tokens) / tokens_per_step))
    max_steps = int(max_steps)
    optimizer = _optimizer(model, config)
    scheduler = _build_scheduler(
        optimizer,
        int(_nested(train_cfg, "warmup_steps", max(1, max_steps // 20))),
        max_steps,
    )
    scaler = make_scaler(device, precision)
    output_dir = Path(_nested(config, "output_dir", "outputs/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    tokens_seen = 0
    best_validation_loss: float | None = None
    latest = load_latest_checkpoint(output_dir, map_location=device) if bool(_nested(train_cfg, "resume", True)) else None
    if latest is not None:
        model.load_state_dict(latest["model"])
        if latest.get("optimizer"):
            optimizer.load_state_dict(latest["optimizer"])
        if latest.get("scheduler"):
            scheduler.load_state_dict(latest["scheduler"])
        if latest.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(latest["scaler"])
        step = int(latest.get("step", 0))
        tokens_seen = int(latest.get("tokens_seen", 0))
        best_validation_loss = latest.get("best_validation_loss")
        if latest.get("rng_state"):
            restore_rng_state(latest["rng_state"])
        print(f"Resumed step={step} tokens_seen={tokens_seen}")

    clip_value = float(_nested(train_cfg, "gradient_clip_norm", 1.0))
    log_interval = int(_nested(train_cfg, "log_interval_steps", 10))
    eval_interval = int(_nested(train_cfg, "eval_interval_steps", 250))
    checkpoint_interval = int(_nested(train_cfg, "checkpoint_interval_steps", eval_interval))
    keep_checkpoints = int(_nested(train_cfg, "keep_checkpoints", 3))
    generation_interval = int(_nested(train_cfg, "generation_interval_steps", eval_interval))
    model.train()
    iterator = iter(loader)
    running_loss = 0.0
    started = time.perf_counter()
    last_validation_loss: float | None = None
    try:
        while step < max_steps and (max_tokens is None or tokens_seen < int(max_tokens)):
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(grad_accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                loss = _batch_loss(model, batch, device, precision)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")
                scaled_loss = loss / grad_accum
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                micro_losses.append(float(loss.detach().cpu()))
                tokens_seen += int(batch["input_ids"].numel())
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = float(clip_grad_norm_(model.parameters(), clip_value).detach().cpu())
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = float(clip_grad_norm_(model.parameters(), clip_value).detach().cpu())
                optimizer.step()
            scheduler.step()
            step += 1
            running_loss += sum(micro_losses) / len(micro_losses)

            if step % log_interval == 0 or step == 1:
                elapsed = max(time.perf_counter() - started, 1e-9)
                throughput = tokens_seen / elapsed
                memory = (
                    torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                )
                print(
                    f"step={step}/{max_steps} loss={running_loss / min(log_interval, step):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3g} tokens={tokens_seen} "
                    f"tok/s={throughput:.1f} grad_norm={grad_norm:.2f} gpu_mem_gb={memory:.2f}"
                )
                running_loss = 0.0

            if val_loader is not None and (step % eval_interval == 0 or step == max_steps):
                last_validation_loss, val_tokens, val_throughput = evaluate_model(
                    model,
                    val_loader,
                    device=device,
                    precision=precision,
                    max_batches=int(_nested(train_cfg, "validation_batches", 50)),
                )
                print(
                    f"validation step={step} loss={last_validation_loss:.4f} "
                    f"ppl={math.exp(min(20.0, last_validation_loss)):.3f} "
                    f"tok/s={val_throughput:.1f} tokens={val_tokens}"
                )
                if best_validation_loss is None or last_validation_loss < best_validation_loss:
                    best_validation_loss = last_validation_loss
                    best_path = save_checkpoint(
                        output_dir,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        tokens_seen=tokens_seen,
                        config={**config, "model": model_config_dict(model)},
                        train_loss=sum(micro_losses) / len(micro_losses),
                        validation_loss=last_validation_loss,
                        best_validation_loss=best_validation_loss,
                        protected=True,
                    )
                    _save_best(best_path, output_dir)

            if step % checkpoint_interval == 0 or step == max_steps:
                save_checkpoint(
                    output_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    tokens_seen=tokens_seen,
                    config={**config, "model": model_config_dict(model)},
                    train_loss=sum(micro_losses) / len(micro_losses),
                    validation_loss=last_validation_loss,
                    best_validation_loss=best_validation_loss,
                )
                prune_checkpoints(output_dir, keep_checkpoints)
            if generation_interval and step % generation_interval == 0:
                print_samples(model, config, device)
    except KeyboardInterrupt:
        print("Interrupted; writing an emergency checkpoint")
        save_checkpoint(
            output_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokens_seen=tokens_seen,
            config={**config, "model": model_config_dict(model)},
            train_loss=None,
            validation_loss=last_validation_loss,
            best_validation_loss=best_validation_loss,
            protected=True,
        )
        raise

    result = {
        "step": step,
        "tokens_seen": tokens_seen,
        "parameters": model.num_parameters(),
        "best_validation_loss": best_validation_loss,
        "output_dir": str(output_dir),
        "device": str(device),
        "precision": precision,
    }
    write_json(output_dir / "training_summary.json", result)
    save_inference_export(
        model,
        output_dir / "inference.pt",
        model_config=model_config_dict(model),
        step=step,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or dry-run the selective SSM language model.")
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument("--dry-run", action="store_true", help="Run forward/backward/optimizer/generation only")
    parser.add_argument("--device", help="Override config device, e.g. cpu or cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml(args.config)
    if args.device:
        config["device"] = args.device
    device = choose_device(str(_nested(config, "device", "auto")))
    precision = select_precision(str(_nested(config, "precision", "auto")), device)
    if args.dry_run:
        dry_run(config, device, precision)
    else:
        print(json.dumps(train(config), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
