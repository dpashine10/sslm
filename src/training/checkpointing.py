"""Atomic, resumable checkpoint helpers.

Checkpoint files are written to a temporary file in the destination directory
and then renamed.  A Kaggle interruption therefore leaves either the previous
valid checkpoint or the new complete checkpoint, never a misleading half-file.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_VERSION = 1
_STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Save a torch payload atomically and return the final path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all visible CUDA RNG states."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    tokens_seen: int,
    config: Mapping[str, Any],
    train_loss: float | None = None,
    validation_loss: float | None = None,
    best_validation_loss: float | None = None,
    protected: bool = False,
) -> Path:
    """Write a complete training checkpoint and a small JSON sidecar."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "protected": bool(protected),
    }
    path = atomic_torch_save(payload, output / f"checkpoint_step_{step:08d}.pt")
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "best_validation_loss": best_validation_loss,
                "protected": protected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def save_inference_export(
    model: torch.nn.Module,
    path: str | Path,
    *,
    model_config: Mapping[str, Any],
    step: int = 0,
) -> Path:
    """Write a weights/config-only artifact without optimizer or RNG state."""

    return atomic_torch_save(
        {
            "format": "ssm-inference-v1",
            "checkpoint_version": CHECKPOINT_VERSION,
            "step": int(step),
            "model": model.state_dict(),
            "model_config": dict(model_config),
        },
        path,
    )


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.6
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model" not in payload or "step" not in payload:
        raise ValueError(f"Invalid or incomplete checkpoint: {path}")
    return payload


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate one complete checkpoint."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return _torch_load(checkpoint, map_location=map_location)


def valid_checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    """Return valid checkpoint files sorted by step, ignoring partial files."""

    candidates: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("checkpoint_step_*.pt"):
        match = _STEP_RE.match(path.name)
        if not match:
            continue
        try:
            _torch_load(path, map_location="cpu")
        except Exception:
            continue
        candidates.append((int(match.group(1)), path))
    return sorted(candidates)


def load_latest_checkpoint(
    output_dir: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any] | None:
    """Load the newest valid checkpoint, or return ``None``."""

    candidates = valid_checkpoints(output_dir)
    if not candidates:
        return None
    return load_checkpoint(candidates[-1][1], map_location=map_location)


def prune_checkpoints(output_dir: str | Path, keep: int = 3) -> list[Path]:
    """Retain recent checkpoints plus best/protected checkpoints."""

    if keep < 1:
        raise ValueError("keep must be positive")
    candidates = valid_checkpoints(output_dir)
    if len(candidates) <= keep:
        return []
    retained = {path for _, path in candidates[-keep:]}
    for _, path in candidates:
        try:
            metadata = _torch_load(path, map_location="cpu")
        except Exception:
            continue
        if metadata.get("protected") or path.name == "best.pt":
            retained.add(path)
    removed: list[Path] = []
    for _, path in candidates:
        if path not in retained:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            removed.append(path)
    return removed
