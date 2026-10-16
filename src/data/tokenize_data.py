"""Encode processed JSONL shards into raw NumPy memory-mappable token files.

Examples
--------
    python -m src.data.tokenize_data \
        --processed-dir data/processed/wiki \
        --tokenizer data/tokenized/wiki/tokenizer.json \
        --output-dir data/tokenized/wiki/tokens

Each ``.bin`` file contains a contiguous native-endian NumPy array with no
header, so it can be opened with ``numpy.memmap`` without copying it into
RAM.  ``uint16`` is selected for vocabularies up to 65,535 entries and
``uint32`` otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import iter_jsonl, write_json


def _split_paths(processed_dir: Path, split: str) -> list[Path]:
    paths = sorted(processed_dir.glob(f"{split}-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No {split}-*.jsonl shards found under {processed_dir}")
    return paths


def _tokenizer_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file does not exist: {path}")
    return path


def tokenize_data(
    processed_dir: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
    add_eos: bool = True,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Encode processed shards and return a token-shard manifest."""

    try:
        import numpy as np
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("tokenize_data.py requires 'numpy' and 'tokenizers'") from exc

    source = Path(processed_dir)
    tokenizer_file = _tokenizer_path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.dtype("uint16" if vocab_size <= 65_535 else "uint32")
    eos_id = tokenizer.token_to_id("<eos>")
    if add_eos and eos_id is None:
        raise ValueError("--add-eos was requested, but tokenizer has no <eos> token")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive when provided")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_manifests: dict[str, list[dict[str, Any]]] = {}
    total_tokens = 0
    total_records = 0

    for split in splits:
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split {split!r}; use train or validation")
        paths = _split_paths(source, split)
        shard_manifests: list[dict[str, Any]] = []
        split_token_count = 0
        for source_path in paths:
            if max_tokens is not None and split_token_count >= max_tokens:
                break
            encoded_records: list[list[int]] = []
            token_count = 0
            for record in iter_jsonl(source_path):
                if max_tokens is not None and split_token_count >= max_tokens:
                    break
                text = record.get("text")
                if not isinstance(text, str):
                    continue
                ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
                if add_eos:
                    ids.append(int(eos_id))
                if max_tokens is not None:
                    remaining = max_tokens - split_token_count
                    ids = ids[:remaining]
                    if not ids:
                        break
                encoded_records.append(ids)
                token_count += len(ids)
                split_token_count += len(ids)

            output_path = destination / f"{source_path.stem}.bin"
            output_path.unlink(missing_ok=True)
            if token_count:
                mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(token_count,))
                cursor = 0
                for ids in encoded_records:
                    if ids:
                        end = cursor + len(ids)
                        mmap[cursor:end] = np.asarray(ids, dtype=dtype)
                        cursor = end
                mmap.flush()
                del mmap
            else:
                output_path.touch()
            item = {
                "path": output_path.name,
                "source": source_path.name,
                "records": len(encoded_records),
                "tokens": token_count,
                "dtype": dtype.name,
                "bytes": token_count * dtype.itemsize,
            }
            shard_manifests.append(item)
            total_tokens += token_count
            total_records += len(encoded_records)
        split_manifests[split] = shard_manifests

    manifest = {
        "format": "mmap-token-shards-v1",
        "source": str(source),
        "tokenizer": str(tokenizer_file),
        "vocab_size": vocab_size,
        "dtype": dtype.name,
        "add_eos": add_eos,
        "max_tokens_per_split": max_tokens,
        "eos_id": eos_id,
        "splits": split_manifests,
        "total_records": total_records,
        "total_tokens": total_tokens,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``tokenize_data.py``."""

    parser = argparse.ArgumentParser(
        description="Encode processed JSONL shards into uint16/uint32 mmap-compatible .bin files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-dir", required=True, help="Directory containing train/validation JSONL shards.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json or its containing directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for .bin shards and manifest.json.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
        help="Splits to encode.",
    )
    parser.add_argument("--no-eos", action="store_true", help="Do not append tokenizer <eos> after each record.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional per-split token budget; the final record is truncated to fit the budget.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tokenization CLI."""

    args = build_parser().parse_args(argv)
    manifest = tokenize_data(
        args.processed_dir,
        args.tokenizer,
        args.output_dir,
        splits=tuple(args.splits),
        add_eos=not args.no_eos,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
