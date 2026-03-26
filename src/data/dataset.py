"""
Dataset Module for Language Modeling

This module implements PyTorch datasets for autoregressive language modeling.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """
    Dataset for autoregressive language modeling.

    In autoregressive language modeling, we predict the next token given previous tokens.
    The dataset creates (input, target) pairs where:
    - input: tokens [0, 1, 2, ..., n-1]
    - target: tokens [1, 2, 3, ..., n]

    This is also called "teacher forcing" during training.

    Args:
        data_path: Path to the data file (JSONL format)
        tokenizer: Tokenizer to use for encoding text
        max_seq_len: Maximum sequence length
        split: Data split ("train", "val", or "test")
        split_ratio: Tuple of (train, val, test) ratios
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.9, 0.05, 0.05),
        split_seed: Optional[int] = 42,
        add_special_tokens: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.split_seed = split_seed
        self.add_special_tokens = add_special_tokens

        # TODO: Load and tokenize data
        print(f"Loading data from {data_path}...")
        self.examples = self._load_and_tokenize()

        # Keep the split deterministic but avoid taking contiguous slices from
        # an ordered corpus (e.g. Tiny Shakespeare), which can bias val/test.
        if self.split_seed is not None:
            rng = random.Random(self.split_seed)
            rng.shuffle(self.examples)

        # TODO: Split data
        # STUDENT TODO: Split examples into train/val/test based on split_ratio
        total = len(self.examples)
        train_size = int(total * split_ratio[0])
        val_size = int(total * split_ratio[1])
        val_end = train_size + val_size

        if split == "train":
            self.examples = self.examples[:train_size]
        elif split == "val":
            self.examples = self.examples[train_size:val_end]
        elif split == "test":
            self.examples = self.examples[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"Loaded {len(self.examples)} examples for {split} split")

    def _load_and_tokenize(self) -> List[List[int]]:
        """
        Load data from file and tokenize.

        Returns:
            List of tokenized examples (each example is a list of token IDs)
        """
        examples = []

        # TODO: Read JSONL file
        # Each line is a JSON object with a "text" field
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                # STUDENT TODO: Parse JSON line
                data = json.loads(line)

                # STUDENT TODO: Get text from data
                text = data.get("text", "").strip()
                if not text:
                    continue

                # TODO: Tokenize text
                # STUDENT TODO: Use self.tokenizer.encode()
                token_ids = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)

                # TODO: Skip if too short or too long
                if len(token_ids) < 2 or len(token_ids) > self.max_seq_len:
                    continue

                examples.append(token_ids)

        return examples

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Index of the example

        Returns:
            Tuple of (input_ids, target_ids)
            - input_ids: tokens [0, 1, 2, ..., n-1]
            - target_ids: tokens [1, 2, 3, ..., n]
        """
        # TODO: Get token IDs for this example
        token_ids = self.examples[idx]

        # TODO: Create input and target sequences
        # For autoregressive LM:
        # input: [BOS, token1, token2, ..., token_n-1]
        # target: [token1, token2, ..., token_n-1, EOS]
        # Or simply:
        # input: token_ids[:-1]
        # target: token_ids[1:]

        # STUDENT TODO: Create input_ids (all tokens except last)
        input_ids = token_ids[:-1]

        # STUDENT TODO: Create target_ids (all tokens except first)
        target_ids = token_ids[1:]

        # TODO: Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        return input_ids, target_ids


class TextDataset(Dataset):
    """
    Simple text dataset that loads all text into memory.

    This is useful for smaller datasets or when you want to process
    the entire dataset at once.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
    """

    def __init__(self, texts: List[str], tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # TODO: Tokenize all texts
        print(f"Tokenizing {len(texts)} texts...")
        self.examples = []
        for text in texts:
            # STUDENT TODO: Tokenize text
            token_ids = self.tokenizer.encode(text)

            # Skip if too short or too long
            if len(token_ids) < 2 or len(token_ids) > max_seq_len:
                continue

            self.examples.append(token_ids)

        print(f"Created {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input and target sequences."""
        token_ids = self.examples[idx]

        # STUDENT TODO: Create input and target (same as LanguageModelingDataset)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


class StreamingDataset(Dataset):
    """
    Streaming dataset that doesn't load all data into memory.

    This is useful for very large datasets that don't fit in memory.
    It reads data on-the-fly during training.

    Args:
        data_path: Path to data file
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
        max_examples: Maximum number of examples to use (None for all)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        max_examples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # TODO: Count lines in file to get dataset size
        # STUDENT TODO: Count number of lines in file
        with open(data_path, "r") as f:
            self.num_examples = sum(1 for _ in f)

        if max_examples is not None:
            self.num_examples = min(self.num_examples, max_examples)

        print(f"Streaming dataset with {self.num_examples} examples")

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get example by reading from file.

        Note: This is inefficient for random access. In practice, you'd want
        to use memory mapping or a database for better performance.
        """
        # TODO: Read the idx-th line from file
        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    # STUDENT TODO: Parse and tokenize this line
                    data = json.loads(line)
                    text = data.get("text", "")
                    token_ids = self.tokenizer.encode(text)
                    break

        # STUDENT TODO: Create input and target
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


# Utility function to prepare data
def prepare_data(
    data_path: str,
    output_path: str,
    tokenizer,
    max_examples: Optional[int] = None,
):
    """
    Prepare and save tokenized data.

    This pre-tokenizes the data and saves it, which speeds up training.

    Args:
        data_path: Path to raw data file
        output_path: Path to save tokenized data
        tokenizer: Tokenizer to use
        max_examples: Maximum number of examples to process
    """
    print(f"Preparing data from {data_path}...")

    tokenized_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            data = json.loads(line)
            text = data["text"]
            token_ids = tokenizer.encode(text)
            tokenized_data.append(token_ids)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1} examples")

    # Save tokenized data
    with open(output_path, "w") as f:
        json.dump(tokenized_data, f)

    print(f"Saved {len(tokenized_data)} tokenized examples to {output_path}")


# Test function
def test_dataset():
    """
    Test dataset implementation.
    """
    from ..tokenizer.base import CharacterTokenizer

    # Create dummy data
    texts = [
        "Hello world, this is a test.",
        "Another example sentence here.",
        "Machine learning is fascinating!",
    ]

    # Create tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)

    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_seq_len=50)

    print(f"Dataset size: {len(dataset)}")

    # Get first example
    input_ids, target_ids = dataset[0]
    print(f"\nFirst example:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Target IDs: {target_ids.tolist()}")

    # Decode
    input_text = tokenizer.decode(input_ids.tolist())
    target_text = tokenizer.decode(target_ids.tolist())
    print(f"Input text: {input_text}")
    print(f"Target text: {target_text}")
