"""
Byte-level BPE tokenizer.

This tokenizer uses the Hugging Face ``tokenizers`` library with a ByteLevel
pre-tokenizer and decoder so punctuation, whitespace, casing, and line breaks
are preserved much more faithfully than the hand-written teaching BPE.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from .base import BaseTokenizer


class ByteLevelBPETokenizer(BaseTokenizer):
    """Wrapper around a Hugging Face byte-level BPE tokenizer."""

    TOKENIZER_TYPE = "byte_bpe"

    def __init__(self):
        super().__init__()
        self.tokenizer: Tokenizer | None = None
        self.token_to_id = {}
        self.id_to_token = {}

    def _find_token_id(self, *candidates: str, default: int | None = None) -> int:
        """
        Find the first token ID that exists in the vocabulary.

        This lets us reuse existing tokenizers whose special token names differ
        from the small teaching tokenizers. For example, MiniMind uses
        ``<|im_start|>`` / ``<|im_end|>`` instead of ``<BOS>`` / ``<EOS>``.
        """
        for token in candidates:
            if token in self.token_to_id:
                return self.token_to_id[token]
        if default is not None:
            return default
        raise KeyError(f"Could not find any of these special tokens: {candidates}")

    def _sync_from_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been initialized")

        vocab = self.tokenizer.get_vocab()
        self.token_to_id = dict(vocab)
        self.id_to_token = {idx: token for token, idx in vocab.items()}
        self.vocab = self.token_to_id
        self.inverse_vocab = self.id_to_token
        self.special_tokens = {
            "pad_token_id": self._find_token_id("<PAD>", "<|endoftext|>", default=0),
            "bos_token_id": self._find_token_id("<BOS>", "<|im_start|>", "<s>", default=1),
            "eos_token_id": self._find_token_id("<EOS>", "<|im_end|>", "</s>", default=2),
            "unk_token_id": self._find_token_id("<UNK>", "<|endoftext|>", "<unk>", default=0),
        }

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        **kwargs,
    ):
        """
        Train a byte-level BPE tokenizer.

        Args:
            texts: Training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum merge frequency
        """
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer
        self._sync_from_tokenizer()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded")

        token_ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded")

        ids = token_ids
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            ids = [token_id for token_id in token_ids if token_id not in special_ids]

        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: str):
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path_obj))

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self._sync_from_tokenizer()
