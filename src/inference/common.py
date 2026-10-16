"""Shared safe loading and tokenization helpers for inference CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:  # Repository-root module invocation.
    from src.models import ModelConfig, SSMLanguageModel
except ImportError:  # pragma: no cover - installed package invocation
    from models import ModelConfig, SSMLanguageModel
try:
    from src.training.checkpointing import load_checkpoint
    from src.training.utils import choose_device
except ImportError:  # pragma: no cover - installed package invocation
    from training.checkpointing import load_checkpoint
    from training.utils import choose_device


def resolve_tokenizer_path(path: str | Path) -> Path:
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file does not exist: {tokenizer_path}. Pass --tokenizer tokenizer.json."
        )
    return tokenizer_path


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    device: str = "auto",
) -> tuple[SSMLanguageModel, Any, torch.device, dict[str, Any]]:
    """Load either a full training checkpoint or an inference export."""

    selected_device = choose_device(device)
    payload = load_checkpoint(checkpoint_path, map_location=selected_device)
    config = payload.get("config", {})
    model_values = payload.get("model_config") or config.get("model") or config
    allowed = set(ModelConfig.__dataclass_fields__)
    try:
        model_config = ModelConfig(**{key: value for key, value in model_values.items() if key in allowed})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint has no compatible ModelConfig: {checkpoint_path}") from exc
    model = SSMLanguageModel(model_config)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights are incompatible with {model_config}: {checkpoint_path}") from exc
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Inference requires the tokenizers package") from exc
    tokenizer = Tokenizer.from_file(str(resolve_tokenizer_path(tokenizer_path)))
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer.get_vocab_size()}) does not match model vocabulary "
            f"({model_config.vocab_size})"
        )
    model.to(selected_device).eval()
    return model, tokenizer, selected_device, payload


def prompt_ids(tokenizer: Any, prompt: str, max_length: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        raise ValueError("Prompt encoded to zero tokens")
    return torch.tensor([ids[-max_length:]], dtype=torch.long, device=device)


def decode(tokenizer: Any, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=False)
