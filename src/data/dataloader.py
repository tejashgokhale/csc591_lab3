"""
DataLoader Utilities for Language Modeling

Required baseline scope in this file:
- `collate_fn`
- `create_dataloader`

Optional extension in this file:
- `BucketBatchSampler`
- `create_bucketed_dataloader`

The optional bucketed path is provided mainly to reduce confusion in the starter
repo: students do not need it for the 3-week baseline, but instructors or
stronger students can still use it.
"""

import random
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.

    This function pads sequences in a batch to the same length and creates
    an attention mask to indicate which positions are padding.

    Args:
        batch: List of (input_ids, target_ids) tuples
        pad_token_id: ID of the padding token

    Returns:
        Tuple of (input_ids, target_ids, attention_mask)
        - input_ids: Padded input sequences of shape (batch_size, max_seq_len)
        - target_ids: Padded target sequences of shape (batch_size, max_seq_len)
        - attention_mask: Mask of shape (batch_size, max_seq_len) with 1 for real tokens, 0 for padding
    """
    # TODO: Separate inputs and targets
    # STUDENT TODO: Extract input_ids and target_ids from batch
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # TODO: Pad sequences to the same length
    # Hint: Use pad_sequence from torch.nn.utils.rnn
    # pad_sequence expects a list of tensors and returns a padded tensor
    # Set batch_first=True to get shape (batch_size, seq_len)
    # STUDENT TODO: Pad input_ids
    input_ids_padded = None  # STUDENT TODO

    # STUDENT TODO: Pad target_ids
    target_ids_padded = None  # STUDENT TODO

    # TODO: Create attention mask
    # 1 for real tokens, 0 for padding
    # STUDENT TODO: Create mask where padding positions are 0
    # Hint: (input_ids_padded != pad_token_id).long()
    attention_mask = None  # STUDENT TODO

    return input_ids_padded, target_ids_padded, attention_mask


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with proper collation for language modeling.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pad_token_id: Padding token ID
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    # TODO: Create collate function with the specified pad_token_id
    # Hint: Use lambda to create a partial function
    collate = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)

    # TODO: Create DataLoader
    # STUDENT TODO: Create DataLoader with appropriate parameters
    dataloader = None  # STUDENT TODO

    return dataloader


class BucketBatchSampler:
    """
    OPTIONAL EXTENSION.

    Batch sampler that groups sequences of similar lengths together.

    This reduces the amount of padding needed and speeds up training.

    Args:
        dataset: Dataset to sample from
        batch_size: Batch size
        shuffle: Whether to shuffle batches
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.lengths = []
        for i in range(len(dataset)):
            length = len(dataset[i][0])
            self.lengths.append((i, length))

        self.lengths.sort(key=lambda x: x[1])

    def __iter__(self):
        """Iterate over batches."""
        indices = [idx for idx, _ in self.lengths]

        if self.shuffle:
            buckets = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]
            for bucket in buckets:
                random.shuffle(bucket)
            random.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_bucketed_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    OPTIONAL EXTENSION.

    Create a DataLoader with bucket batching for efficient padding.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pad_token_id: Padding token ID
        pin_memory: Whether to pin memory

    Returns:
        DataLoader with bucket batching
    """
    batch_sampler = BucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    collate = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
    )

    return dataloader


# Utility function to compute dataset statistics
def compute_dataset_stats(dataset: Dataset) -> dict:
    """
    Compute statistics about the dataset.

    This helps students understand their data and choose appropriate
    hyperparameters (e.g., max_seq_len, batch_size).

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary with statistics
    """
    lengths = []
    for i in range(len(dataset)):
        input_ids, _ = dataset[i]
        lengths.append(len(input_ids))

    stats = {
        "num_examples": len(dataset),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }

    # Compute percentiles
    sorted_lengths = sorted(lengths)
    for p in [50, 75, 90, 95, 99]:
        idx = int(len(sorted_lengths) * p / 100)
        stats[f"p{p}_length"] = sorted_lengths[idx]

    return stats


def print_dataset_stats(dataset: Dataset):
    """
    Print dataset statistics in a readable format.

    Args:
        dataset: Dataset to analyze
    """
    stats = compute_dataset_stats(dataset)

    print("Dataset Statistics:")
    print("=" * 50)
    print(f"Number of examples: {stats['num_examples']:,}")
    print(f"Sequence length:")
    print(f"  Min: {stats['min_length']}")
    print(f"  Max: {stats['max_length']}")
    print(f"  Mean: {stats['mean_length']:.1f}")
    print(f"  Median: {stats['median_length']}")
    print(f"Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {stats[f'p{p}_length']}")
    print("=" * 50)


# Test function
def test_dataloader():
    """
    Test dataloader implementation.
    """
    from ..tokenizer.base import CharacterTokenizer
    from .dataset import TextDataset

    # Create dummy data
    texts = [
        "Short text.",
        "This is a medium length text example.",
        "Here is another example that is quite a bit longer than the others.",
        "Tiny.",
    ]

    # Create tokenizer and dataset
    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)
    dataset = TextDataset(texts, tokenizer, max_seq_len=100)

    print("Testing DataLoader:")
    print(f"Dataset size: {len(dataset)}\n")

    # Print dataset stats
    print_dataset_stats(dataset)

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(f"\nDataLoader with {len(dataloader)} batches\n")

    # Test one batch
    for batch_idx, (input_ids, target_ids, attention_mask) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Mask shape: {attention_mask.shape}")
        print(f"  Input IDs:\n{input_ids}")
        print(f"  Attention mask:\n{attention_mask}")
        if batch_idx == 0:
            break
