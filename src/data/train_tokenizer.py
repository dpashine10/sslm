"""Train a BPE tokenizer using only processed training shards.

Examples
--------
    python -m src.data.train_tokenizer \
        --processed-dir data/processed/wiki \
        --output-dir data/tokenized/wiki \
        --vocab-size 32000

The validation shards are never read.  The tokenizer is saved as
``tokenizer.json`` and accompanied by a manifest containing its vocabulary and
special-token IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from .common import iter_jsonl, write_json


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _train_paths(processed_dir: Path) -> list[Path]:
    if processed_dir.is_file():
        return [processed_dir]
    paths = sorted(processed_dir.glob("train-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No train-*.jsonl shards found under {processed_dir}")
    return paths


def _text_iterator(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        for record in iter_jsonl(path):
            text = record.get("text")
            if isinstance(text, str) and text:
                yield text


def train_tokenizer(
    processed_dir: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
) -> dict[str, Any]:
    """Train and save a ByteLevel BPE tokenizer from training text only."""

    if vocab_size < len(special_tokens):
        raise ValueError("vocab_size must be at least the number of special tokens")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    if "<unk>" not in special_tokens:
        raise ValueError("special_tokens must include <unk> for the BPE unknown token")
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("train_tokenizer.py requires the 'tokenizers' package") from exc

    source = Path(processed_dir)
    paths = _train_paths(source)
    record_count = 0
    for path in paths:
        record_count += sum(1 for record in iter_jsonl(path) if isinstance(record.get("text"), str) and record["text"])
    if record_count == 0:
        raise ValueError("No non-empty text records found in the training shards")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_text_iterator(paths), trainer=trainer)
    tokenizer_path = destination / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    manifest = {
        "format": "bpe-tokenizer-v1",
        "source": str(source),
        "train_files": [path.name for path in paths],
        "train_records": record_count,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "special_tokens": list(special_tokens),
        "special_token_ids": {
            token: tokenizer.token_to_id(token) for token in special_tokens
        },
        "tokenizer_file": tokenizer_path.name,
    }
    write_json(destination / "tokenizer_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``train_tokenizer.py``."""

    parser = argparse.ArgumentParser(
        description="Train a ByteLevel BPE tokenizer from processed train shards only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-dir", required=True, help="Directory containing train-*.jsonl shards.")
    parser.add_argument("--output-dir", required=True, help="Directory for tokenizer.json and its manifest.")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Maximum BPE vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency for learned merges.")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        help="Special token; repeat to override the defaults (<pad>, <unk>, <bos>, <eos>).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the tokenizer-training CLI."""

    args = build_parser().parse_args(argv)
    manifest = train_tokenizer(
        args.processed_dir,
        args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=tuple(args.special_tokens or SPECIAL_TOKENS),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
