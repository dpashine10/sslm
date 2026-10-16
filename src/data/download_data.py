"""Download a Hugging Face dataset as raw JSONL with schema metadata.

Examples
--------
    python -m src.data.download_data \
        --dataset HuggingFaceH4/ultrachat_200k --split train \
        --output-dir data/raw/ultrachat

The command uses Hugging Face streaming mode, so it does not materialize a
``datasets.Dataset`` in memory.  A failed attempt is discarded before the
next retry.  The final directory contains ``raw.jsonl`` and ``metadata.json``
with the dataset arguments, sample schema, and record count.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import write_json
from .schemas import detect_schema


def _load_stream(
    dataset: str,
    *,
    config: str | None,
    split: str,
    revision: str | None,
) -> Iterable[Mapping[str, Any]]:
    """Load one streaming dataset iterator, importing ``datasets`` lazily."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("download_data.py requires the 'datasets' package") from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if revision:
        kwargs["revision"] = revision
    if config:
        return load_dataset(dataset, config, **kwargs)
    return load_dataset(dataset, **kwargs)


def download_dataset(
    dataset: str,
    output_dir: str | Path,
    *,
    config: str | None = None,
    split: str = "train",
    revision: str | None = None,
    max_records: int | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    sample_size: int = 32,
    license_name: str | None = None,
) -> dict[str, Any]:
    """Stream *dataset* to JSONL and return the written metadata.

    ``retries`` is the total number of attempts, not the number of retries
    after the first attempt.  A temporary file is used so callers never see a
    partial ``raw.jsonl`` after an unsuccessful attempt.
    """

    if retries < 1:
        raise ValueError("retries must be at least 1")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when provided")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "raw.jsonl"
    temporary_path = destination / "raw.jsonl.part"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            temporary_path.unlink(missing_ok=True)
            count = 0
            sample: list[dict[str, Any]] = []
            with temporary_path.open("w", encoding="utf-8") as handle:
                stream = _load_stream(dataset, config=config, split=split, revision=revision)
                for record in stream:
                    if not isinstance(record, Mapping):
                        raise ValueError(f"Dataset yielded a non-object record: {type(record).__name__}")
                    materialized = dict(record)
                    handle.write(json.dumps(materialized, ensure_ascii=False, default=str) + "\n")
                    count += 1
                    if len(sample) < sample_size:
                        sample.append(materialized)
                    if max_records is not None and count >= max_records:
                        break
            temporary_path.replace(output_path)
            metadata = {
                "format": "jsonl",
                "dataset": dataset,
                "config": config,
                "split": split,
                "revision": revision,
                "license": license_name,
                "license_note": "Review the dataset card; pass --license to record the reviewed identifier.",
                "streaming": True,
                "record_count": count,
                "sample_size": len(sample),
                "schema": detect_schema(sample),
                "max_records": max_records,
                "attempts": attempt,
            }
            write_json(destination / "metadata.json", metadata)
            return metadata
        except Exception as exc:  # retry transient network/iterator failures
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt >= retries:
                raise RuntimeError(
                    f"Unable to stream {dataset!r} after {retries} attempt(s): {exc}"
                ) from exc
            if retry_delay:
                time.sleep(retry_delay * (2 ** (attempt - 1)))

    raise RuntimeError(f"Unable to stream {dataset!r}: {last_error}")  # pragma: no cover


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``download_data.py``."""

    parser = argparse.ArgumentParser(
        description="Stream a Hugging Face dataset to raw JSONL and write schema metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset identifier.")
    parser.add_argument("--config", help="Optional dataset configuration/subset name.")
    parser.add_argument("--split", default="train", help="Dataset split to stream.")
    parser.add_argument("--revision", help="Optional dataset branch, tag, or commit.")
    parser.add_argument("--output-dir", required=True, help="Directory for raw.jsonl and metadata.json.")
    parser.add_argument("--max-records", type=int, help="Stop after this many records; useful for smoke tests.")
    parser.add_argument("--retries", type=int, default=3, help="Total streaming attempts.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Initial retry delay in seconds; doubles per attempt.")
    parser.add_argument("--sample-size", type=int, default=32, help="Number of records used for schema detection.")
    parser.add_argument(
        "--license",
        dest="license_name",
        help="License identifier copied into metadata after you review the current dataset card.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the download CLI."""

    args = build_parser().parse_args(argv)
    metadata = download_dataset(
        args.dataset,
        args.output_dir,
        config=args.config,
        split=args.split,
        revision=args.revision,
        max_records=args.max_records,
        retries=args.retries,
        retry_delay=args.retry_delay,
        sample_size=args.sample_size,
        license_name=args.license_name,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
