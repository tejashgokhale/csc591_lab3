"""Minimal SFT formatting helpers for optional Part 3 discussion."""

from __future__ import annotations


def format_sft_prompt(instruction: str, input_text: str = "") -> str:
    """Format a tiny instruction/input prefix for decoder-only SFT examples."""
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    return f"Instruction: {instruction}\nResponse:"
