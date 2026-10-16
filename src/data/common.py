"""Shared, dependency-light helpers for the data pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def json_default(value: Any) -> Any:
    """Convert common non-JSON values into stable JSON-compatible values."""

    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, human-readable JSON to *path*."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, rejecting malformed lines."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            yield value


def iter_jsonl_paths(path: str | Path) -> list[Path]:
    """Return stable JSONL inputs for a file or directory."""

    source = Path(path)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    paths = sorted(
        candidate
        for candidate in source.glob("*.jsonl")
        if candidate.is_file() and "manifest" not in candidate.name.lower()
    )
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found under {source}")
    return paths


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file or all JSONL files in a directory."""

    for jsonl_path in iter_jsonl_paths(path):
        yield from iter_jsonl(jsonl_path)


def clean_text(value: Any) -> str:
    """Normalize Unicode and whitespace while preserving readable content."""

    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    for paragraph in PARAGRAPH_RE.split(value):
        lines = [WHITESPACE_RE.sub(" ", line).strip() for line in paragraph.split("\n")]
        normalized = " ".join(line for line in lines if line)
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def canonical_json(value: Any) -> str:
    """Serialize a value for stable hashing and duplicate detection."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def stable_hash(value: str, seed: str = "") -> str:
    """Return a stable SHA-256 hex digest."""

    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def validation_bucket(value: str, validation_ratio: float, seed: str = "") -> bool:
    """Return whether a value belongs in validation for a deterministic split."""

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in the range [0, 1)")
    if validation_ratio == 0:
        return False
    bucket = int(stable_hash(value, seed)[:16], 16) / float(16**16)
    return bucket < validation_ratio


def shard_name(split: str, index: int, suffix: str = ".jsonl") -> str:
    """Build a stable zero-padded shard name."""

    return f"{split}-{index:05d}{suffix}"


def percentile(values: Iterable[int], fraction: float) -> float:
    """Calculate a small dependency-free nearest-rank percentile."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[position])


def increment(counter: Counter[str], key: str, amount: int = 1) -> None:
    """Increment a Counter with a concise, typed helper."""

    counter[key] += amount


def ensure_mapping(value: Any, name: str = "record") -> Mapping[str, Any]:
    """Validate a mapping-like record and provide a useful error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
