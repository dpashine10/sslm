"""Clean, deduplicate, split, and shard raw JSONL language-model data.

Examples
--------
    python -m src.data.prepare_data \
        --input-dir data/raw/wiki \
        --output-dir data/processed/wiki \
        --validation-ratio 0.01 --shard-size 100000

Each output record is a JSON object with a normalized ``text`` field.  The
split is based on a SHA-256 hash of that text and seed, so rerunning the
command produces the same assignment independent of input ordering.  The
output manifest includes filtering, duplicate, split, and shard statistics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from .common import (
    clean_text,
    iter_jsonl_paths,
    percentile,
    read_json,
    stable_hash,
    validation_bucket,
    write_json,
)
from .schemas import record_to_text


class _ShardWriter:
    """Write one split to bounded JSONL shards."""

    def __init__(self, directory: Path, split: str, shard_size: int) -> None:
        self.directory = directory
        self.split = split
        self.shard_size = shard_size
        self.index = -1
        self.count_in_shard = 0
        self.handle = None
        self.shards: list[dict[str, Any]] = []
        self._open_next()

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.shards[-1]["records"] = self.count_in_shard
        self.index += 1
        path = self.directory / f"{self.split}-{self.index:05d}.jsonl"
        self.handle = path.open("w", encoding="utf-8")
        self.count_in_shard = 0
        self.shards.append({"path": path.name, "records": 0})

    def write(self, record: Mapping[str, Any]) -> None:
        if self.count_in_shard >= self.shard_size:
            self._open_next()
        assert self.handle is not None
        self.handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        self.count_in_shard += 1

    def close(self) -> list[dict[str, Any]]:
        if self.handle is not None:
            self.handle.close()
            self.shards[-1]["records"] = self.count_in_shard
            self.handle = None
        if self.shards and self.shards[-1]["records"] == 0:
            self.shards.pop()
            (self.directory / f"{self.split}-{self.index:05d}.jsonl").unlink(missing_ok=True)
        return self.shards


def _iter_jsonl_tolerant(paths: list[Path], stats: Counter[str]) -> Iterator[Mapping[str, Any]]:
    """Yield valid object records while counting malformed input lines."""

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    stats["blank_lines"] += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_records"] += 1
                    continue
                if not isinstance(record, Mapping):
                    stats["non_object_records"] += 1
                    continue
                yield record


def prepare_data(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.01,
    seed: str = "0",
    min_chars: int = 1,
    max_chars: int | None = None,
    shard_size: int = 100_000,
    dedupe: bool = True,
) -> dict[str, Any]:
    """Prepare generic text data and return its manifest."""

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in the range [0, 1)")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")
    if max_chars is not None and max_chars < min_chars:
        raise ValueError("max_chars must be greater than or equal to min_chars")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")

    source = Path(input_path)
    paths = iter_jsonl_paths(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        for old_shard in destination.glob(f"{split}-*.jsonl"):
            old_shard.unlink()

    stats: Counter[str] = Counter()
    lengths: list[int] = []
    seen: set[str] = set()
    writers = {
        "train": _ShardWriter(destination, "train", shard_size),
        "validation": _ShardWriter(destination, "validation", shard_size),
    }
    try:
        for record in _iter_jsonl_tolerant(paths, stats):
            stats["input_records"] += 1
            text = clean_text(record_to_text(record))
            if not text:
                stats["filtered_empty"] += 1
                continue
            if len(text) < min_chars:
                stats["filtered_too_short"] += 1
                continue
            if max_chars is not None and len(text) > max_chars:
                stats["filtered_too_long"] += 1
                continue
            digest = stable_hash(text, seed)
            if dedupe and digest in seen:
                stats["duplicates"] += 1
                continue
            seen.add(digest)
            split = "validation" if validation_bucket(text, validation_ratio, seed) else "train"
            writers[split].write({"text": text})
            stats[split + "_records"] += 1
            stats[split + "_characters"] += len(text)
            lengths.append(len(text))
    finally:
        shard_manifest = {split: writers[split].close() for split in writers}

    source_metadata: dict[str, Any] | None = None
    if source.is_dir() and (source / "metadata.json").exists():
        try:
            source_metadata = read_json(source / "metadata.json")
        except (OSError, json.JSONDecodeError):
            source_metadata = None

    manifest = {
        "format": "processed-jsonl-v1",
        "input": str(source),
        "input_files": [path.name for path in paths],
        "output_dir": str(destination),
        "config": {
            "validation_ratio": validation_ratio,
            "seed": seed,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "shard_size": shard_size,
            "dedupe": dedupe,
        },
        "stats": {
            **dict(stats),
            "accepted_records": stats["train_records"] + stats["validation_records"],
            "character_percentiles": {
                "p50": percentile(lengths, 0.50),
                "p95": percentile(lengths, 0.95),
                "p99": percentile(lengths, 0.99),
            },
        },
        "shards": shard_manifest,
    }
    if source_metadata is not None:
        manifest["source_metadata"] = source_metadata
    write_json(destination / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``prepare_data.py``."""

    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, deterministically split, and shard JSONL text data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "--input-dir", dest="input_path", required=True, help="Raw JSONL file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed shards and manifest.json.")
    parser.add_argument("--validation-ratio", type=float, default=0.01, help="Fraction assigned to validation by hash.")
    parser.add_argument("--seed", default="0", help="String mixed into the deterministic split and dedupe hash.")
    parser.add_argument("--min-chars", type=int, default=1, help="Minimum cleaned text length.")
    parser.add_argument("--max-chars", type=int, help="Maximum cleaned text length; longer records are dropped.")
    parser.add_argument("--shard-size", type=int, default=100_000, help="Maximum records per output shard.")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate cleaned records.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the preparation CLI."""

    args = build_parser().parse_args(argv)
    manifest = prepare_data(
        args.input_path,
        args.output_dir,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        shard_size=args.shard_size,
        dedupe=not args.no_dedupe,
    )
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
