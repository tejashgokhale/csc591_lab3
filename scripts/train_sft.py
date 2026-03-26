#!/usr/bin/env python3
"""
Optional SFT stub.

The starter repository currently treats Part 3 as a design/report extension.
This script exists so users get a clear message instead of a missing-file error.
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optional SFT entry point. This starter repo does not ship a full "
            "end-to-end SFT training scaffold."
        )
    )
    parser.add_argument("--config", type=str, help="Reference-only optional SFT config path")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        help="Optional base checkpoint path for your own experimentation",
    )
    parser.parse_args()

    print(
        "This starter repo does not currently include a full released SFT trainer.\n"
        "Treat SFT as an optional design/report extension for now.\n"
        "You may still use scripts/download_sft_dataset.py to inspect a small SFT-style dataset."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
