"""Efficient random-window loading from token ``.bin`` shards.

``RandomWindowDataset`` keeps token shards as read-only NumPy memory maps and
only copies one language-model window per ``__getitem__`` call.  A map-style
dataset works directly with ``torch.utils.data.DataLoader(..., shuffle=True)``
without loading a complete split into RAM.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

try:  # Keep metadata and preparation modules importable without PyTorch.
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _TorchDataset = object  # type: ignore[assignment,misc]


class RandomWindowDataset(_TorchDataset):
    """Map a token manifest split to fixed-length next-token windows.

    Parameters
    ----------
    manifest:
        A tokenization ``manifest.json`` path or its containing directory.
    split:
        ``train`` or ``validation``.
    block_size:
        Number of input tokens returned.  Each sample's labels are the next
        ``block_size`` tokens in the same shard.

    Windows never cross shard boundaries.  This keeps random access cheap and
    avoids silently joining unrelated independently written files.
    """

    def __init__(self, manifest: str | Path, split: str = "train", block_size: int = 1024) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency declaration covers normal use
            raise RuntimeError("RandomWindowDataset requires the 'numpy' package") from exc

        manifest_path = Path(manifest)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Token manifest does not exist: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        split_items = self._split_items(metadata, split)
        if not split_items:
            raise ValueError(f"Token manifest contains no shards for split {split!r}")

        self.manifest_path = manifest_path
        self.split = split
        self.block_size = block_size
        self._arrays: list[Any] = []
        self._window_ends: list[int] = []
        self._tokens = 0
        for item in split_items:
            relative = Path(str(item["path"]))
            shard_path = relative if relative.is_absolute() else manifest_path.parent / relative
            if not shard_path.is_file():
                raise FileNotFoundError(f"Token shard does not exist: {shard_path}")
            dtype = np.dtype(str(item.get("dtype", metadata.get("dtype", "uint16"))))
            if dtype not in (np.dtype("uint16"), np.dtype("uint32")):
                raise ValueError(f"Unsupported token dtype {dtype} in {shard_path}")
            expected_tokens = int(item.get("tokens", shard_path.stat().st_size // dtype.itemsize))
            file_bytes = shard_path.stat().st_size
            if file_bytes % dtype.itemsize:
                raise ValueError(f"Shard byte size is not aligned to {dtype.name}: {shard_path}")
            actual_tokens = file_bytes // dtype.itemsize
            if actual_tokens != expected_tokens:
                raise ValueError(
                    f"Shard size mismatch for {shard_path}: manifest says {expected_tokens}, file has {actual_tokens}"
                )
            if actual_tokens == 0:
                continue
            array = np.memmap(shard_path, dtype=dtype, mode="r", shape=(actual_tokens,))
            self._arrays.append(array)
            self._tokens += actual_tokens
            self._window_ends.append(self._window_ends[-1] if self._window_ends else 0)
            self._window_ends[-1] += max(0, actual_tokens - block_size)

        self._total_windows = self._window_ends[-1] if self._window_ends else 0

    @staticmethod
    def _split_items(metadata: dict[str, Any], split: str) -> list[dict[str, Any]]:
        splits = metadata.get("splits", {})
        if isinstance(splits, dict):
            items = splits.get(split, [])
            if isinstance(items, list):
                return items
        # Accept a simple manifest variant with a top-level shard list.
        if split == "train" and isinstance(metadata.get("shards"), list):
            return metadata["shards"]
        return []

    def __len__(self) -> int:
        """Return the number of non-crossing next-token windows."""

        return self._total_windows

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"window index out of range: {index}")
        shard_index = bisect.bisect_right(self._window_ends, index)
        previous_end = self._window_ends[shard_index - 1] if shard_index else 0
        return shard_index, index - previous_end

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return ``input_ids`` and one-token-shifted ``labels``."""

        import numpy as np

        shard_index, local_start = self._locate(index)
        values = self._arrays[shard_index]
        input_ids = np.asarray(values[local_start : local_start + self.block_size], dtype=np.int64).copy()
        labels = np.asarray(values[local_start + 1 : local_start + self.block_size + 1], dtype=np.int64).copy()
        try:
            import torch
        except ImportError:  # pragma: no cover - only minimal environments
            return {"input_ids": input_ids, "labels": labels}
        return {"input_ids": torch.from_numpy(input_ids), "labels": torch.from_numpy(labels)}

    @property
    def token_count(self) -> int:
        """Number of tokens in the selected split."""

        return self._tokens


# Short aliases make the loader convenient for existing training code without
# duplicating implementation or creating a second memory-map policy.
TokenWindowDataset = RandomWindowDataset
MemmapDataset = RandomWindowDataset

__all__ = ["RandomWindowDataset", "TokenWindowDataset", "MemmapDataset"]
