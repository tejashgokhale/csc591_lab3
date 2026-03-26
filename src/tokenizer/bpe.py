"""
Byte Pair Encoding (BPE) Tokenizer

This module implements a BPE tokenizer for subword tokenization.
BPE is a data compression technique adapted for tokenization that iteratively
merges the most frequent pairs of tokens.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .base import BaseTokenizer
from tqdm import tqdm


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) Tokenizer.

    BPE algorithm:
    1. Start with a vocabulary of individual characters
    2. Iteratively merge the most frequent pair of tokens
    3. Repeat until desired vocabulary size is reached

    This creates a subword vocabulary that balances between character-level
    and word-level tokenization.

    Advantages:
    - Handles unknown words by breaking them into subwords
    - More efficient than character-level (shorter sequences)
    - Captures common subword patterns

    Used in: GPT-2, GPT-3, RoBERTa, and many other models
    """

    def __init__(self):
        super().__init__()
        self.merges = {}  # Merge rules: (token1, token2) -> merged_token
        self.merge_priority = {}  # Track order of merges: (token1, token2) -> priority
        self.token_to_id = {}
        self.id_to_token = {}

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        **kwargs,
    ):
        """
        Train BPE tokenizer on a corpus.

        Args:
            texts: List of text strings to train on
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a pair to be merged
        """
        print(f"Training BPE tokenizer with target vocab_size={vocab_size}")

        # Initialize vocabulary with special tokens
        vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.special_tokens = {
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 3,
        }

        # Pre-tokenize texts into words
        # Use simple whitespace splitting or regex
        # We'll add a special end-of-word token </w> to mark word boundaries
        print("Pre-tokenizing texts...")
        word_freqs = Counter()

        for text in tqdm(texts, desc="Pre-tokenizing", unit="texts"):
            # Split text into words and count frequencies
            words = re.findall(r'\w+', text.lower())
            for word in words:
                # Add </w> to mark end of word
                word_freqs[word] += 1

        print(f"Found {len(word_freqs)} unique words")

        # Initialize vocabulary with all characters
        print("Building initial character vocabulary...")
        # Collect all unique characters from words
        chars = set()
        for word in word_freqs.keys():
            # Add characters from word to chars set
            for char in word:
                chars.add(char)
        # Add end-of-word token
        chars.add("</w>")

        # Add characters to vocabulary
        vocab.extend(sorted(chars))

        # Convert words to character sequences
        # Each word becomes a list of characters with </w> at the end
        word_splits = {}
        for word, freq in word_freqs.items():
            # Split word into characters
            # Example: "hello" -> ["h", "e", "l", "l", "o", "</w>"]
            word_splits[word] = list(word) + ["</w>"]

        # Iteratively merge most frequent pairs
        print(f"Learning {vocab_size - len(vocab)} merges...")
        num_merges = vocab_size - len(vocab)

        for merge_idx in tqdm(range(num_merges), desc="Learning merges", unit="merge"):
            # Count all adjacent pairs in all words
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = word_splits[word]
                # Count pairs in this word
                # Iterate through adjacent tokens in split
                for j in range(len(split) - 1):
                    pair = (split[j], split[j+1])
                    pair_freqs[pair] += freq

            # Find the most frequent pair
            if not pair_freqs:
                break

            # Get the pair with maximum frequency
            best_pair = max(pair_freqs, key=pair_freqs.get)

            # Check minimum frequency
            if pair_freqs[best_pair] < min_frequency:
                break

            # Merge the best pair in all words
            # Create merged token by concatenating the pair
            merged_token = best_pair[0] + best_pair[1]

            # Add to vocabulary and merges
            vocab.append(merged_token)
            self.merges[best_pair] = merged_token
            self.merge_priority[best_pair] = merge_idx  # Track order of learning

            # Update word splits with the merge
            for word in word_splits:
                split = word_splits[word]
                # Replace all occurrences of best_pair with merged_token
                # Iterate through split and merge adjacent pairs
                new_split = []
                j = 0
                while j < len(split):
                    # Check if current and next token form best_pair
                    # If yes, add merged_token and skip both
                    # If no, add current token and move to next
                    if j < len(split) - 1 and (split[j], split[j+1]) == best_pair:
                        new_split.append(merged_token)
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                word_splits[word] = new_split

        # Create token to ID mapping
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        self.vocab = self.token_to_id
        self.inverse_vocab = self.id_to_token

        print(f"Training complete. Final vocab size: {len(vocab)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges.

        Args:
            word: Word to tokenize

        Returns:
            List of subword tokens
        """
        # Start with character-level split
        # Split word into characters and add </w>
        tokens = list(word) + ["</w>"]

        # Apply merges iteratively
        # Keep merging until no more merges can be applied
        while len(tokens) > 1:
            # Find pairs in current tokens
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]

            # Find the pair with the earliest merge (lowest priority number = learned first)
            # Check which pairs exist in self.merges
            # Choose the one that was learned first (has lowest priority value)
            mergeable_pairs = [pair for pair in pairs if pair in self.merges]

            if not mergeable_pairs:
                break

            # Merge the best pair
            # Find the pair that was learned first (has lowest priority value)
            best_pair = min(mergeable_pairs, key=lambda p: self.merge_priority[p])

            # Replace pair with merged token
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs using BPE.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Pre-tokenize into words
        # Split text into words
        words = re.findall(r'\w+', text.lower())

        # Tokenize each word with BPE
        tokens = []
        for word in words:
            # Tokenize word and add to tokens
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            # Get token ID, use unk_token_id if not found
            token_id = self.token_to_id.get(token, self.special_tokens["unk_token_id"])
            token_ids.append(token_id)

        # Add special tokens if requested
        if add_special_tokens:
            # Add BOS and EOS
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
        # Convert IDs to tokens
        tokens = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()

        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in special_ids:
                continue

            # Convert ID to token
            token = self.id_to_token.get(token_id, "<UNK>")
            tokens.append(token)

        # Join tokens and remove </w> markers
        # Join tokens and replace </w> with spaces
        text = "".join(tokens).replace("</w>", " ")

        return text.strip()

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "merge_priority": {f"{k[0]}|||{k[1]}": v for k, v in self.merge_priority.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.merges = {
            tuple(k.split("|||")): v for k, v in data["merges"].items()
        }
        self.merge_priority = {
            tuple(k.split("|||")): v for k, v in data.get("merge_priority", {}).items()
        }
        self.special_tokens = data["special_tokens"]
        self.vocab = self.token_to_id
        self.inverse_vocab = self.id_to_token
        print(f"Tokenizer loaded from {path}")


# Utility function to use HuggingFace tokenizers library
def create_hf_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 8000,
    save_path: str = None,
) -> "BPETokenizer":
    """
    Create a BPE tokenizer using HuggingFace tokenizers library.

    This is provided as an alternative for students who want to use a
    production-ready tokenizer implementation.

    Args:
        texts: List of texts to train on
        vocab_size: Target vocabulary size
        save_path: Optional path to save the tokenizer

    Returns:
        BPETokenizer wrapper around HuggingFace tokenizer
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
    except ImportError:
        raise ImportError(
            "HuggingFace tokenizers library not installed. "
            "Install with: pip install tokenizers"
        )

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
    )

    # Create iterator from texts
    def batch_iterator(batch_size=1000):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    if save_path:
        tokenizer.save(save_path)

    # Wrap in our BPETokenizer interface
    # (This is simplified; students would need to implement full wrapper)
    return tokenizer


# # Test function
# def test_bpe_tokenizer():
#     """
#     Test BPE tokenizer implementation.
#     """
#     # Sample texts
#     texts = [
#         "hello world",
#         "hello there",
#         "world peace",
#         "hello hello world",
#     ]

#     print("Testing BPE Tokenizer:")
#     print(f"Training on {len(texts)} texts\n")

#     # Train tokenizer
#     tokenizer = BPETokenizer()
#     tokenizer.train(texts, vocab_size=50)

#     # Test encoding
#     test_text = "hello world"
#     token_ids = tokenizer.encode(test_text)
#     print(f"Text: '{test_text}'")
#     print(f"Token IDs: {token_ids}")

#     # Test decoding
#     decoded = tokenizer.decode(token_ids)
#     print(f"Decoded: '{decoded}'")

#     # Test with unknown word
#     test_text2 = "hello universe"
#     token_ids2 = tokenizer.encode(test_text2)
#     print(f"\nText: '{test_text2}'")
#     print(f"Token IDs: {token_ids2}")
#     decoded2 = tokenizer.decode(token_ids2)
#     print(f"Decoded: '{decoded2}'")

#     print(f"\nVocabulary size: {tokenizer.vocab_size}")
