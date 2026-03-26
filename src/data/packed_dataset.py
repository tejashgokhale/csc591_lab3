"""
Low-memory packed dataset utilities.

The idea is simple:
- tokenize once
- save all token IDs in one flat binary file
- save offsets/lengths for each example
- train by memory-mapping the saved arrays

This avoids keeping millions of Python lists of token IDs in RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def is_packed_dataset_dir(data_path: str | Path) -> bool:
    """Return True if a directory looks like a packed dataset."""
    data_dir = Path(data_path)
    return data_dir.is_dir() and (data_dir / "metadata.json").exists()


class PackedTokenDataset(Dataset):
    """
    Dataset backed by memory-mapped token files.

    Each example is already tokenized and stored as:
    - a slice inside one flat token array
    - an offset + length entry

    This keeps memory usage much lower than storing `list[list[int]]`.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

        if not is_packed_dataset_dir(self.data_dir):
            raise ValueError(f"{data_dir} is not a packed dataset directory")

        with open(self.data_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        token_dtype = np.dtype(self.metadata["token_dtype"])
        total_tokens = int(self.metadata["total_tokens"])

        if int(self.metadata["num_examples"]) == 0 or total_tokens == 0:
            raise ValueError(
                f"Packed dataset at {self.data_dir} is empty. "
                "Please prepare it again with a larger max_seq_len or more examples."
            )

        self.tokens = np.memmap(
            self.data_dir / "tokens.bin",
            dtype=token_dtype,
            mode="r",
            shape=(total_tokens,),
        )
        self.offsets = np.load(self.data_dir / "offsets.npy", mmap_mode="r")
        self.lengths = np.load(self.data_dir / "lengths.npy", mmap_mode="r")
        self.indices = np.load(self.data_dir / f"{split}_idx.npy", mmap_mode="r")

        print(
            f"Loaded packed dataset from {self.data_dir} "
            f"({len(self.indices)} {split} examples)"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        example_idx = int(self.indices[idx])
        start = int(self.offsets[example_idx])
        length = int(self.lengths[example_idx])

        token_ids = np.asarray(self.tokens[start : start + length], dtype=np.int64)

        input_ids = torch.from_numpy(token_ids[:-1].copy()).long()
        target_ids = torch.from_numpy(token_ids[1:].copy()).long()
        return input_ids, target_ids
