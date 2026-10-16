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


def _load_viewer_rows(
    dataset: str,
    *,
    config: str | None,
    split: str,
    max_records: int | None,
    max_tokens: int | None,
    page_size: int = 100,
    page_retries: int = 5,
    timeout: float = 120.0,
    page_delay: float = 0.05,
) -> Iterable[Mapping[str, Any]]:
    """Read public rows through the Hugging Face Dataset Viewer API.

    The viewer API is useful on networks where the dataset's signed object
    storage hostname is unavailable. It is capped at 100 rows per request and
    does not expose a revision parameter, so callers should record the API
    backend and run date in the manifest. The regular ``datasets`` backend is
    still preferred when it works.
    """

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - project dependency normally provides it
        raise RuntimeError("viewer_api backend requires requests") from exc
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be in [1, 100]")
    base_url = "https://datasets-server.huggingface.co/rows"
    session = requests.Session()
    offset = 0
    records = 0
    tokens = 0
    while True:
        params = {
            "dataset": dataset,
            "config": config or "default",
            "split": split,
            "offset": offset,
            "length": page_size,
        }
        last_error: Exception | None = None
        for attempt in range(page_retries):
            try:
                response = session.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("rows")
                if not isinstance(rows, list):
                    raise ValueError(f"Dataset Viewer response has no rows list: {payload}")
                break
            except Exception as exc:  # network, JSON, and API errors are retryable per page
                last_error = exc
                if attempt + 1 >= page_retries:
                    raise RuntimeError(
                        f"Dataset Viewer request failed at offset {offset} after {page_retries} attempts: {exc}"
                    ) from exc
                retry_after = response.headers.get("Retry-After") if "response" in locals() else None
                try:
                    delay = float(retry_after) if retry_after else min(60.0, 2**attempt)
                except ValueError:
                    delay = min(60.0, 2**attempt)
                time.sleep(delay)
        else:  # pragma: no cover
            raise RuntimeError(f"Dataset Viewer request failed: {last_error}")
        if not rows:
            break
        for item in rows:
            record = item.get("row") if isinstance(item, Mapping) else None
            if not isinstance(record, Mapping):
                continue
            materialized = dict(record)
            yield materialized
            records += 1
            record_tokens = materialized.get("token_count")
            if not isinstance(record_tokens, (int, float)) or record_tokens <= 0:
                text = materialized.get("text", "")
                record_tokens = max(1, round(len(text) / 4)) if isinstance(text, str) else 1
            tokens += int(record_tokens)
            if max_records is not None and records >= max_records:
                return
            if max_tokens is not None and tokens >= max_tokens:
                return
        offset += len(rows)
        if page_delay:
            time.sleep(page_delay)
        if len(rows) < page_size:
            break


def download_dataset(
    dataset: str,
    output_dir: str | Path,
    *,
    config: str | None = None,
    split: str = "train",
    revision: str | None = None,
    max_records: int | None = None,
    max_tokens: int | None = None,
    retries: int = 3,
    retry_delay: float = 2.0,
    sample_size: int = 32,
    license_name: str | None = None,
    backend: str = "datasets",
    page_size: int = 100,
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
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive when provided")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if backend not in {"datasets", "viewer_api"}:
        raise ValueError("backend must be datasets or viewer_api")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "raw.jsonl"
    temporary_path = destination / "raw.jsonl.part"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            temporary_path.unlink(missing_ok=True)
            count = 0
            token_count = 0
            sample: list[dict[str, Any]] = []
            with temporary_path.open("w", encoding="utf-8") as handle:
                if backend == "viewer_api":
                    stream = _load_viewer_rows(
                        dataset,
                        config=config,
                        split=split,
                        max_records=max_records,
                        max_tokens=max_tokens,
                        page_size=page_size,
                    )
                else:
                    stream = _load_stream(dataset, config=config, split=split, revision=revision)
                for record in stream:
                    if not isinstance(record, Mapping):
                        raise ValueError(f"Dataset yielded a non-object record: {type(record).__name__}")
                    materialized = dict(record)
                    handle.write(json.dumps(materialized, ensure_ascii=False, default=str) + "\n")
                    count += 1
                    record_tokens = materialized.get("token_count")
                    if not isinstance(record_tokens, (int, float)) or record_tokens <= 0:
                        text = materialized.get("text", "")
                        record_tokens = max(1, round(len(text) / 4)) if isinstance(text, str) else 1
                    token_count += int(record_tokens)
                    if len(sample) < sample_size:
                        sample.append(materialized)
                    if max_records is not None and count >= max_records:
                        break
                    if max_tokens is not None and token_count >= max_tokens:
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
                "backend": backend,
                "viewer_api_revision_note": (
                    "Dataset Viewer API served current data; revision is not enforced by this backend."
                    if backend == "viewer_api"
                    else None
                ),
                "streaming": True,
                "record_count": count,
                "sample_size": len(sample),
                "schema": detect_schema(sample),
                "max_records": max_records,
                "max_tokens": max_tokens,
                "token_count_from_records": token_count,
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
    parser.add_argument(
        "--backend",
        choices=("datasets", "viewer_api"),
        default="datasets",
        help="Streaming backend. viewer_api avoids signed object-storage downloads on restricted networks.",
    )
    parser.add_argument("--page-size", type=int, default=100, help="Viewer API rows per request (maximum 100).")
    parser.add_argument("--output-dir", required=True, help="Directory for raw.jsonl and metadata.json.")
    parser.add_argument("--max-records", type=int, help="Stop after this many records; useful for smoke tests.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Stop after approximately this many source token_count values; falls back to characters/4.",
    )
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
        max_tokens=args.max_tokens,
        retries=args.retries,
        retry_delay=args.retry_delay,
        sample_size=args.sample_size,
        license_name=args.license_name,
        backend=args.backend,
        page_size=args.page_size,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
