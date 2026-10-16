"""Normalize Dolly/OASST-style conversations for supervised fine-tuning.

Examples
--------
    python -m src.data.prepare_sft \
        --input data/raw/dolly/raw.jsonl \
        --output-dir data/processed/sft \
        --tokenizer data/tokenized/base/tokenizer.json \
        --max-length 2048

The command accepts ``instruction``/``context``/``response`` records and
``messages`` or ``conversations`` arrays containing ``role``/``content`` or
``from``/``value`` entries.  It emits normalized role names, token IDs, and a
per-token ``loss_mask`` whose value is one only for assistant turns.  Without
``--tokenizer`` it still emits normalized messages and a rendered ``text``
field, which is useful for inspecting a new source schema.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from .common import canonical_json, iter_jsonl_paths, percentile, stable_hash, write_json
from .schemas import record_to_messages, render_messages, validate_messages


def _iter_jsonl_tolerant(paths: list[Path], stats: Counter[str]) -> Iterator[Mapping[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    stats["blank_lines"] += 1
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_json"] += 1
                    continue
                if not isinstance(record, Mapping):
                    stats["non_object_records"] += 1
                    continue
                yield record


def _load_tokenizer(path: str | Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("prepare_sft.py requires the 'tokenizers' package when --tokenizer is used") from exc
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file does not exist: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path)), tokenizer_path


def _encode_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int | None,
    truncate_side: str,
    add_eos: bool,
) -> tuple[list[int], list[int], dict[str, int | bool]]:
    """Encode each turn separately so assistant loss masks stay aligned."""

    input_ids: list[int] = []
    loss_mask: list[int] = []
    for message in messages:
        segment = f'<|{message["role"]}|>\n{message["content"]}\n'
        ids = list(tokenizer.encode(segment, add_special_tokens=False).ids)
        input_ids.extend(ids)
        loss_mask.extend([1 if message["role"] == "assistant" else 0] * len(ids))
    if add_eos:
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is not None:
            input_ids.append(int(eos_id))
            loss_mask.append(1 if messages[-1]["role"] == "assistant" else 0)

    before = len(input_ids)
    assistant_before = sum(loss_mask)
    truncated = before > max_length if max_length is not None else False
    if max_length is not None and truncated:
        if truncate_side == "left":
            input_ids = input_ids[-max_length:]
            loss_mask = loss_mask[-max_length:]
        else:
            input_ids = input_ids[:max_length]
            loss_mask = loss_mask[:max_length]
    return input_ids, loss_mask, {
        "tokens_before": before,
        "tokens_after": len(input_ids),
        "tokens_truncated": before - len(input_ids),
        "assistant_tokens_before": assistant_before,
        "assistant_tokens_after": sum(loss_mask),
        "truncated": truncated,
    }


def prepare_sft(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    max_length: int | None = 2048,
    truncate_side: str = "left",
    shard_size: int = 10_000,
    dedupe: bool = True,
    add_eos: bool = True,
) -> dict[str, Any]:
    """Prepare normalized SFT records and return a statistics manifest."""

    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be positive when provided")
    if truncate_side not in {"left", "right"}:
        raise ValueError("truncate_side must be 'left' or 'right'")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")

    tokenizer = None
    resolved_tokenizer_path: Path | None = None
    if tokenizer_path is not None:
        tokenizer, resolved_tokenizer_path = _load_tokenizer(tokenizer_path)

    source = Path(input_path)
    paths = iter_jsonl_paths(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for old_shard in destination.glob("sft-*.jsonl"):
        old_shard.unlink()

    stats: Counter[str] = Counter()
    seen: set[str] = set()
    token_lengths: list[int] = []
    assistant_lengths: list[int] = []
    output_handle = None
    shard_index = -1
    shard_count = 0
    shard_manifest: list[dict[str, Any]] = []

    def open_shard() -> None:
        nonlocal output_handle, shard_index, shard_count
        if output_handle is not None:
            output_handle.close()
            shard_manifest[-1]["records"] = shard_count
        shard_index += 1
        output_handle = (destination / f"sft-{shard_index:05d}.jsonl").open("w", encoding="utf-8")
        shard_count = 0
        shard_manifest.append({"path": f"sft-{shard_index:05d}.jsonl", "records": 0})

    open_shard()
    try:
        for record in _iter_jsonl_tolerant(paths, stats):
            stats["input_records"] += 1
            messages = record_to_messages(record)
            valid, reason = validate_messages(messages)
            if not valid:
                stats[reason or "malformed_record"] += 1
                continue
            digest = stable_hash(canonical_json(messages))
            if dedupe and digest in seen:
                stats["duplicates"] += 1
                continue
            seen.add(digest)

            output_record: dict[str, Any] = {
                "messages": messages,
                "text": render_messages(messages),
            }
            if tokenizer is not None:
                ids, mask, truncation = _encode_messages(
                    tokenizer,
                    messages,
                    max_length=max_length,
                    truncate_side=truncate_side,
                    add_eos=add_eos,
                )
                output_record.update(
                    {
                        "input_ids": ids,
                        "loss_mask": mask,
                        "num_tokens": len(ids),
                        "num_assistant_tokens": sum(mask),
                        "truncated": truncation["truncated"],
                    }
                )
                for key, value in truncation.items():
                    if key != "truncated":
                        stats[key] += int(value)
                token_lengths.append(len(ids))
                assistant_lengths.append(sum(mask))
                if truncation["truncated"]:
                    stats["truncated_records"] += 1

            assert output_handle is not None
            if shard_count >= shard_size:
                open_shard()
            output_handle.write(json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n")
            shard_count += 1
            stats["accepted_records"] += 1
    finally:
        if output_handle is not None:
            output_handle.close()
            shard_manifest[-1]["records"] = shard_count
        if shard_manifest and shard_manifest[-1]["records"] == 0:
            shard_manifest.pop()
            (destination / f"sft-{shard_index:05d}.jsonl").unlink(missing_ok=True)

    manifest = {
        "format": "sft-jsonl-v1",
        "input": str(source),
        "input_files": [path.name for path in paths],
        "output_dir": str(destination),
        "tokenizer": str(resolved_tokenizer_path) if resolved_tokenizer_path else None,
        "assistant_only_loss_mask": tokenizer is not None,
        "config": {
            "max_length": max_length,
            "truncate_side": truncate_side,
            "shard_size": shard_size,
            "dedupe": dedupe,
            "add_eos": add_eos,
        },
        "stats": {
            **dict(stats),
            "token_percentiles": {
                "p50": percentile(token_lengths, 0.50),
                "p95": percentile(token_lengths, 0.95),
                "p99": percentile(token_lengths, 0.99),
            },
            "assistant_token_percentiles": {
                "p50": percentile(assistant_lengths, 0.50),
                "p95": percentile(assistant_lengths, 0.95),
                "p99": percentile(assistant_lengths, 0.99),
            },
        },
        "shards": shard_manifest,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``prepare_sft.py``."""

    parser = argparse.ArgumentParser(
        description="Normalize Dolly/OASST conversations and optionally create assistant-only token masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "--input-dir", dest="input_path", required=True, help="Raw JSONL file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for sft-*.jsonl and manifest.json.")
    parser.add_argument("--tokenizer", help="tokenizer.json or directory containing it; enables input_ids/loss_mask.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum encoded sequence length.")
    parser.add_argument("--truncate-side", choices=("left", "right"), default="left", help="Side removed when a sequence is too long.")
    parser.add_argument("--shard-size", type=int, default=10_000, help="Maximum records per SFT shard.")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate normalized conversations.")
    parser.add_argument("--no-eos", action="store_true", help="Do not append <eos> when the tokenizer has one.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the SFT preparation CLI."""

    args = build_parser().parse_args(argv)
    manifest = prepare_sft(
        args.input_path,
        args.output_dir,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
        truncate_side=args.truncate_side,
        shard_size=args.shard_size,
        dedupe=not args.no_dedupe,
        add_eos=not args.no_eos,
    )
    print(json.dumps(manifest["stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
