"""Shared configuration, device, and model-building helpers."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:  # Works both from the repository root and from an installed package.
    from src.models import ModelConfig, SSMLanguageModel
except ImportError:  # pragma: no cover - installed console scripts
    from models import ModelConfig, SSMLanguageModel


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("YAML configuration requires PyYAML") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def select_precision(requested: str, device: torch.device) -> str:
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if requested == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp16"
        return "fp32"
    if requested != "fp32" and device.type not in {"cuda", "mps"}:
        print(f"Requested {requested} on {device.type}; falling back to fp32")
        return "fp32"
    return requested


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def infer_vocab_size(token_manifest: str | Path) -> int:
    with (Path(token_manifest) / "manifest.json" if Path(token_manifest).is_dir() else Path(token_manifest)).open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    value = manifest.get("vocab_size")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Token manifest must contain a positive integer vocab_size")
    return value


def build_model(config: Mapping[str, Any], *, vocab_size: int | None = None) -> SSMLanguageModel:
    values = dict(config.get("model", config))
    if values.get("vocab_size") in {None, "auto"}:
        if vocab_size is None:
            raise ValueError("model.vocab_size is unset; provide a token manifest")
        values["vocab_size"] = vocab_size
    allowed = set(ModelConfig.__dataclass_fields__)
    model_values = {key: value for key, value in values.items() if key in allowed}
    return SSMLanguageModel(ModelConfig(**model_values))


def model_config_dict(model: SSMLanguageModel) -> dict[str, Any]:
    return asdict(model.config)


def environment_report(device: torch.device, precision: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "precision": precision,
    }
    if device.type == "cuda":
        report.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "gpu_vram_gb": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2),
                "cuda": torch.version.cuda,
            }
        )
    else:
        report["gpu"] = None
    return report
