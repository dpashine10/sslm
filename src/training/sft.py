"""Optional assistant-only supervised fine-tuning from prepared JSONL records.

``prepare_sft.py`` creates ``input_ids`` and ``loss_mask``.  This trainer sets
non-assistant target positions to ``ignore_index`` while retaining all prompt
tokens as recurrent context; it never mixes SFT examples into pretraining.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:  # Repository-root module invocation.
    from src.models import CausalLMOutput, ModelConfig, SSMLanguageModel
except ImportError:  # pragma: no cover - installed package invocation
    from models import CausalLMOutput, ModelConfig, SSMLanguageModel
from .checkpointing import load_checkpoint, prune_checkpoints, save_checkpoint
from .train import _build_scheduler
from .utils import autocast_context, choose_device, load_yaml, make_scaler, seed_everything, select_precision


class PreparedSFTDataset(Dataset):
    """Load prepared token/mask records without requiring a Hugging Face runtime."""

    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        paths = [source] if source.is_file() else sorted(source.glob("sft-*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No sft-*.jsonl files found under {source}")
        self.records: list[tuple[list[int], list[int]]] = []
        for shard in paths:
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ids = record.get("input_ids")
                    mask = record.get("loss_mask")
                    if (
                        isinstance(ids, list)
                        and isinstance(mask, list)
                        and len(ids) == len(mask) >= 2
                        and any(int(value) for value in mask)
                    ):
                        self.records.append(([int(value) for value in ids], [int(value) for value in mask]))
        if not self.records:
            raise ValueError("SFT data contains no tokenized records with assistant targets")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.records[index]


def _collate(batch: list[tuple[list[int], list[int]]], pad_id: int, ignore_index: int) -> dict[str, torch.Tensor]:
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), ignore_index, dtype=torch.long)
    for row, (ids, mask) in enumerate(batch):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        targets = torch.tensor(ids, dtype=torch.long)
        target_mask = torch.tensor(mask, dtype=torch.bool)
        targets[~target_mask] = ignore_index
        labels[row, :length] = targets
    return {"input_ids": input_ids, "labels": labels}


def _load_model(checkpoint: str | Path, device: torch.device) -> tuple[SSMLanguageModel, dict[str, Any]]:
    payload = load_checkpoint(checkpoint, map_location=device)
    values = payload.get("model_config") or payload.get("config", {}).get("model")
    if not isinstance(values, dict):
        raise ValueError("Pretraining checkpoint does not contain model configuration")
    allowed = set(ModelConfig.__dataclass_fields__)
    model = SSMLanguageModel(ModelConfig(**{key: value for key, value in values.items() if key in allowed}))
    model.load_state_dict(payload["model"])
    return model.to(device), payload


def train_sft(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config.get("seed", 1234)))
    device = choose_device(str(config.get("device", "auto")))
    precision = select_precision(str(config.get("precision", "auto")), device)
    train_cfg = config.get("training", {})
    checkpoint = config.get("pretrained_checkpoint")
    data_path = config.get("sft_data")
    if not checkpoint or not data_path:
        raise ValueError("pretrained_checkpoint and sft_data are required")
    model, source_payload = _load_model(checkpoint, device)
    dataset = PreparedSFTDataset(data_path)
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda rows: _collate(rows, model.config.pad_token_id or 0, model.config.ignore_index),
    )
    max_steps = int(train_cfg.get("max_steps", 1000))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-5)),
        betas=(0.9, 0.95),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler = _build_scheduler(optimizer, int(train_cfg.get("warmup_steps", 50)), max_steps)
    scaler = make_scaler(device, precision)
    output_dir = Path(config.get("output_dir", "outputs/checkpoints/sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    if bool(train_cfg.get("resume", True)):
        latest = None
        from .checkpointing import load_latest_checkpoint

        latest = load_latest_checkpoint(output_dir, map_location=device)
        if latest:
            model.load_state_dict(latest["model"])
            optimizer.load_state_dict(latest["optimizer"])
            if latest.get("scheduler"):
                scheduler.load_state_dict(latest["scheduler"])
            if latest.get("scaler") and scaler.is_enabled():
                scaler.load_state_dict(latest["scaler"])
            step = int(latest.get("step", 0))
    model.train()
    iterator = iter(loader)
    started = time.perf_counter()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        tokens = 0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast_context(device, precision):
                output = model(input_ids, targets=labels)
            if not isinstance(output, CausalLMOutput) or output.loss is None:
                raise RuntimeError("SFT model did not return a loss")
            if not torch.isfinite(output.loss):
                raise FloatingPointError("non-finite SFT loss")
            if scaler.is_enabled():
                scaler.scale(output.loss / grad_accum).backward()
            else:
                (output.loss / grad_accum).backward()
            losses.append(float(output.loss.detach().cpu()))
            tokens += int(input_ids.numel())
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), float(train_cfg.get("gradient_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), float(train_cfg.get("gradient_clip_norm", 1.0)))
            optimizer.step()
        scheduler.step()
        step += 1
        if step == 1 or step % int(train_cfg.get("log_interval_steps", 10)) == 0:
            print(f"sft_step={step}/{max_steps} loss={sum(losses) / len(losses):.4f} tokens={tokens}")
        if step % int(train_cfg.get("checkpoint_interval_steps", 100)) == 0 or step == max_steps:
            save_checkpoint(
                output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tokens_seen=step * tokens,
                config={**config, "model": model.config.__dict__, "source_step": source_payload.get("step")},
                train_loss=sum(losses) / len(losses),
            )
            prune_checkpoints(output_dir, int(train_cfg.get("keep_checkpoints", 2)))
    result = {
        "step": step,
        "parameters": model.num_parameters(),
        "seconds": time.perf_counter() - started,
        "output_dir": str(output_dir),
        "device": str(device),
        "precision": precision,
    }
    (output_dir / "sft_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional assistant-only SFT trainer.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(train_sft(load_yaml(args.config)), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
