"""
Base Tokenizer Interface

This module defines the base interface for tokenizers.
"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    A tokenizer converts between text and token IDs. It handles:
    - Encoding: text -> token IDs
    - Decoding: token IDs -> text
    - Special tokens (PAD, BOS, EOS, UNK)
    - Vocabulary management
    """

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {}

    @abstractmethod
    def train(self, texts: List[str], vocab_size: int, **kwargs):
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: List of text strings to train on
            vocab_size: Target vocabulary size
            **kwargs: Additional training arguments
        """
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save tokenizer to file.

        Args:
            path: Path to save the tokenizer
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load tokenizer from file.

        Args:
            path: Path to load the tokenizer from
        """
        pass

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.special_tokens.get("pad_token_id", 0)

    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.special_tokens.get("bos_token_id", 1)

    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.special_tokens.get("eos_token_id", 2)

    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.special_tokens.get("unk_token_id", 3)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of text strings
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_special_tokens) for text in texts]

    def decode_batch(
        self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            token_ids_batch: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_ids_batch]


class CharacterTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer.

    This tokenizer treats each character as a token. It's simple but results in
    long sequences and doesn't capture subword information.

    This is provided as a simple baseline for students to understand tokenization.
    """

    def __init__(self):
        super().__init__()
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, texts: List[str], vocab_size: int = None, **kwargs):
        """
        Train character tokenizer by building vocabulary from texts.

        Args:
            texts: List of text strings
            vocab_size: Not used for character tokenizer (all chars are included)
        """
        # Collect all unique characters from texts
        chars = set()
        for text in texts:
            for char in text:
                chars.add(char)

        # Add special tokens to the beginning of vocabulary
        special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

        # Create character to ID mapping
        # Special tokens get IDs 0-3, then sorted characters
        vocab_list = special_tokens + sorted(chars)
        self.char_to_id = {char: idx for idx, char in enumerate(vocab_list)}
        self.id_to_char = {idx: char for idx, char in enumerate(vocab_list)}

        # Store special token IDs
        self.special_tokens = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 3,
        }

        self.vocab = self.char_to_id
        self.inverse_vocab = self.id_to_char

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            List of token IDs
        """
        # Convert each character to its ID
        # Use unk_token_id for unknown characters
        token_ids = []
        for char in text:
            token_id = self.char_to_id.get(char, self.special_tokens["unk_token_id"])
            token_ids.append(token_id)

        # Add special tokens if requested
        if add_special_tokens:
            token_ids = [self.special_tokens["bos_token_id"]] + token_ids + [self.special_tokens["eos_token_id"]]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        # Convert token IDs back to characters
        chars = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()

        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in special_ids:
                continue

            # Convert ID to character and append to chars
            char = self.id_to_char.get(token_id, "<UNK>")
            chars.append(char)

        return "".join(chars)

    def save(self, path: str):
        """Save tokenizer to file."""
        import json

        data = {
            "char_to_id": self.char_to_id,
            "id_to_char": {int(k): v for k, v in self.id_to_char.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Load tokenizer from file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self.special_tokens = data["special_tokens"]
        self.vocab = self.char_to_id
        self.inverse_vocab = self.id_to_char
