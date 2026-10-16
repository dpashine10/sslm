"""Evaluate a checkpoint on a tokenized validation split."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:  # Repository-root module invocation.
    from src.data.dataset import RandomWindowDataset
    from src.models import ModelConfig, SSMLanguageModel
except ImportError:  # pragma: no cover - installed package invocation
    from data.dataset import RandomWindowDataset
    from models import ModelConfig, SSMLanguageModel
from .checkpointing import load_checkpoint
from .train import _batch_loss
from .utils import autocast_context, choose_device, environment_report, select_precision, write_json


def _load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[SSMLanguageModel, dict[str, Any]]:
    payload = load_checkpoint(checkpoint_path, map_location=device)
    config = payload.get("config", {})
    model_values = config.get("model", config)
    allowed = set(ModelConfig.__dataclass_fields__)
    model = SSMLanguageModel(ModelConfig(**{k: v for k, v in model_values.items() if k in allowed}))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    token_manifest: str | Path,
    *,
    split: str = "validation",
    block_size: int | None = None,
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
    precision: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    selected_device = choose_device(device)
    model, payload = _load_model(checkpoint_path, selected_device)
    selected_precision = select_precision(precision, selected_device)
    block_size = block_size or model.config.max_seq_len
    dataset = RandomWindowDataset(token_manifest, split=split, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    losses: list[float] = []
    tokens = 0
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        loss = _batch_loss(model, batch, selected_device, selected_precision)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at validation batch {index}")
        losses.append(float(loss.cpu()))
        tokens += int(batch["input_ids"].numel())
    elapsed = max(time.perf_counter() - started, 1e-9)
    loss = sum(losses) / max(1, len(losses))
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": payload.get("step"),
        "split": split,
        "batches": len(losses),
        "tokens_processed": tokens,
        "loss": loss,
        "perplexity": math.exp(min(20.0, loss)),
        "tokens_per_second": tokens / elapsed,
        "parameters": model.num_parameters(),
        "environment": environment_report(selected_device, selected_precision),
    }
    if selected_device.type == "cuda":
        report["peak_memory_gb"] = torch.cuda.max_memory_allocated(selected_device) / 2**30
    if output_path:
        write_json(output_path, report)
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an SSM LM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--token-manifest", required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_checkpoint(
        args.checkpoint,
        args.token_manifest,
        split=args.split,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        precision=args.precision,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
