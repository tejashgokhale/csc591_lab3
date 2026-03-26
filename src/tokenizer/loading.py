"""Tokenizer loading helpers used by scripts and tests."""

from __future__ import annotations

import json

from .base import CharacterTokenizer
from .bpe import BPETokenizer
from .byte_bpe import ByteLevelBPETokenizer


def detect_tokenizer_type(tokenizer_path: str) -> str:
    """Detect which tokenizer implementation should load a JSON file."""
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "char_to_id" in data:
        return "char"
    if (
        data.get("model", {}).get("type") == "BPE"
        and data.get("pre_tokenizer", {}).get("type") == "ByteLevel"
    ):
        return "byte_bpe"
    if "token_to_id" in data or "merges" in data:
        return "bpe"
    raise ValueError(f"Unknown tokenizer format in {tokenizer_path}")


def load_tokenizer(tokenizer_path: str):
    """Load a tokenizer file produced by the lab scripts."""
    tokenizer_type = detect_tokenizer_type(tokenizer_path)

    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer()
    elif tokenizer_type == "byte_bpe":
        tokenizer = ByteLevelBPETokenizer()
    else:
        tokenizer = BPETokenizer()

    tokenizer.load(tokenizer_path)
    return tokenizer
