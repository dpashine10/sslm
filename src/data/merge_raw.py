"""Merge raw JSONL dataset downloads while retaining source provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import write_json
from .schemas import detect_schema


def merge_raw(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_root = Path(input_dir)
    files = sorted(path for path in source_root.glob("**/raw.jsonl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No raw.jsonl files found below {source_root}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "raw.jsonl"
    sample: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    total = 0
    with output_path.open("w", encoding="utf-8") as output:
        for source in files:
            count = 0
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(value, dict):
                        continue
                    output.write(json.dumps(value, ensure_ascii=False) + "\n")
                    count += 1
                    total += 1
                    if len(sample) < 32:
                        sample.append(value)
            counts[str(source.relative_to(source_root))] = count
    source_metadata = []
    for raw_file in files:
        metadata_file = raw_file.with_name("metadata.json")
        if metadata_file.is_file():
            try:
                source_metadata.append(json.loads(metadata_file.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                source_metadata.append({"metadata_file": str(metadata_file)})
    manifest = {
        "format": "merged-raw-jsonl-v1",
        "input_dir": str(source_root),
        "output": output_path.name,
        "source_files": [str(path.relative_to(source_root)) for path in files],
        "source_record_counts": counts,
        "record_count": total,
        "schema": detect_schema(sample),
        "source_metadata": source_metadata,
    }
    write_json(destination / "metadata.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge downloaded raw.jsonl files and preserve metadata.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(merge_raw(args.input_dir, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
