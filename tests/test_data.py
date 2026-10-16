"""Offline tests for the data and tokenizer pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.download_data import download_dataset
from src.data.common import clean_text
from src.data.prepare_data import is_obvious_boilerplate, prepare_data
from src.data.prepare_sft import prepare_sft
from src.data.schemas import detect_schema, normalize_messages


FIXTURES = Path(__file__).parent / "fixtures"


def _jsonl_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def test_schema_detection_and_message_normalization() -> None:
    records = _jsonl_records(FIXTURES / "sft_records.jsonl")[:3]
    schema = detect_schema(records)
    assert schema["kind"] == "conversation"
    messages = normalize_messages(
        [
            {"from": "human", "value": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    assert messages == [
        {"role": "user", "content": "first\n\nsecond"},
        {"role": "assistant", "content": "answer"},
    ]


def test_clean_text_preserves_paragraph_boundaries() -> None:
    assert clean_text(" first line\nsecond line\n\n  next paragraph \n") == (
        "first line second line\n\nnext paragraph"
    )


def test_conservative_boilerplate_filter() -> None:
    assert is_obvious_boilerplate("Please enable JavaScript to continue. Cookie policy.")
    assert not is_obvious_boilerplate("A useful article about the history of computers.")


def test_download_writes_raw_jsonl_metadata_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.data.download_data as download_module

    rows = _jsonl_records(FIXTURES / "raw_records.jsonl")[:4]
    attempts = {"count": 0}

    def fake_stream(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("temporary local fixture failure")
        return iter(rows)

    monkeypatch.setattr(download_module, "_load_stream", fake_stream)
    metadata = download_dataset(
        "local-fixture",
        tmp_path,
        retries=2,
        retry_delay=0,
        sample_size=4,
    )
    assert attempts["count"] == 2
    assert metadata["record_count"] == 4
    assert metadata["token_count_from_records"] > 0
    assert metadata["schema"]["kind"] == "conversation"
    assert len((tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["attempts"] == 2


def test_prepare_data_is_deterministic_and_filters_duplicates(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    source = FIXTURES / "raw_records.jsonl"
    first = prepare_data(source, first_dir, validation_ratio=0.4, seed="fixture", min_chars=3, shard_size=1)
    second = prepare_data(source, second_dir, validation_ratio=0.4, seed="fixture", min_chars=3, shard_size=1)

    assert first["stats"] == second["stats"]
    assert first["stats"]["duplicates"] == 1
    assert first["stats"]["malformed_records"] == 1
    assert first["stats"]["non_object_records"] == 1
    assert first["stats"]["filtered_empty"] == 1
    assert first["stats"]["filtered_too_short"] == 1
    assert sorted(first["shards"]) == ["train", "validation"]
    first_records = sorted(first_dir.glob("*.jsonl"))
    second_records = sorted(second_dir.glob("*.jsonl"))
    assert [path.name for path in first_records] == [path.name for path in second_records]
    assert [path.read_text(encoding="utf-8") for path in first_records] == [
        path.read_text(encoding="utf-8") for path in second_records
    ]


def test_tokenizer_tokenization_and_random_windows_are_mmap_backed(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    np = pytest.importorskip("numpy")

    from src.data.dataset import RandomWindowDataset
    from src.data.tokenize_data import tokenize_data
    from src.data.train_tokenizer import train_tokenizer

    processed_dir = tmp_path / "processed"
    prepare_data(FIXTURES / "raw_records.jsonl", processed_dir, validation_ratio=0, min_chars=3, shard_size=2)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_manifest = train_tokenizer(
        processed_dir,
        tokenizer_dir,
        vocab_size=128,
        min_frequency=1,
    )
    assert tokenizer_manifest["train_files"]
    assert tokenizer_manifest["special_token_ids"]["<eos>"] is not None

    token_dir = tmp_path / "tokens"
    token_manifest = tokenize_data(
        processed_dir,
        tokenizer_dir,
        token_dir,
        splits=("train",),
    )
    assert token_manifest["dtype"] in {"uint16", "uint32"}
    token_file = token_dir / token_manifest["splits"]["train"][0]["path"]
    mapped = np.memmap(token_file, dtype=token_manifest["dtype"], mode="r")
    assert len(mapped) == token_manifest["splits"]["train"][0]["tokens"]

    dataset = RandomWindowDataset(token_dir, split="train", block_size=2)
    assert len(dataset) > 0
    sample = dataset[0]
    assert tuple(sample["input_ids"].shape) == (2,)
    assert tuple(sample["labels"].shape) == (2,)


def test_prepare_sft_normalizes_masks_and_reports_truncation(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    from src.data.tokenize_data import tokenize_data
    from src.data.train_tokenizer import train_tokenizer

    processed_dir = tmp_path / "processed"
    prepare_data(FIXTURES / "raw_records.jsonl", processed_dir, validation_ratio=0, min_chars=3)
    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer(processed_dir, tokenizer_dir, vocab_size=128, min_frequency=1)

    output_dir = tmp_path / "sft"
    manifest = prepare_sft(
        FIXTURES / "sft_records.jsonl",
        output_dir,
        tokenizer_path=tokenizer_dir,
        max_length=8,
        shard_size=2,
    )
    assert manifest["assistant_only_loss_mask"] is True
    assert manifest["stats"]["duplicates"] == 1
    assert manifest["stats"]["malformed_json"] == 1
    assert manifest["stats"]["missing_assistant"] == 1
    assert manifest["stats"]["truncated_records"] > 0
    records = _jsonl_records(output_dir / manifest["shards"][0]["path"])
    assert records
    for record in records:
        assert len(record["input_ids"]) <= 8
        assert len(record["input_ids"]) == len(record["loss_mask"])
        assert 1 in record["loss_mask"]
