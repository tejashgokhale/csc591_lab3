#!/usr/bin/env python3
"""
Download a small sample dataset and convert it into the JSONL format expected by
the lab codebase.

This script is intentionally lightweight: it only uses the Python standard
library so students do not need to learn an additional dataset toolchain before
they can start working with the training pipeline.

Example:
    python scripts/download_sample_dataset.py \
        --dataset tinyshakespeare \
        --output_path output/tinyshakespeare/tinyshakespeare.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlopen


DATASET_SOURCES = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "description": "Tiny Shakespeare dialogue corpus",
    }
}


def download_text(url: str) -> str:
    """Download UTF-8 text from a URL."""
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def group_lines(lines: List[str], group_size: int) -> Iterable[str]:
    """
    Group non-empty lines together.

    Grouping a few lines per JSONL record makes the examples less fragmented
    while still keeping them very small and easy to inspect.
    """
    buffer: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        buffer.append(line)
        if len(buffer) == group_size:
            yield " ".join(buffer)
            buffer = []

    if buffer:
        yield " ".join(buffer)


def write_jsonl(records: Iterable[str], output_path: Path, source_name: str):
    """Write records to JSONL with a single required `text` field."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, text in enumerate(records):
            record = {
                "text": text,
                "source": source_name,
                "example_id": idx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Download a small sample dataset for the lab")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_SOURCES.keys()),
        default="tinyshakespeare",
        help="Which sample dataset to download",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/tinyshakespeare/tinyshakespeare.jsonl",
        help="Where to save the converted JSONL dataset",
    )
    parser.add_argument(
        "--group_lines",
        type=int,
        default=4,
        help="How many non-empty raw lines to merge into one JSONL example",
    )
    args = parser.parse_args()

    if args.group_lines <= 0:
        raise ValueError("--group_lines must be a positive integer")

    source = DATASET_SOURCES[args.dataset]
    output_path = Path(args.output_path)

    print("=" * 80)
    print(f"Downloading sample dataset: {args.dataset}")
    print(f"Description: {source['description']}")
    print(f"Source URL: {source['url']}")
    print(f"Output path: {output_path}")
    print(f"Grouping {args.group_lines} non-empty lines per example")
    print("=" * 80)

    raw_text = download_text(source["url"])
    raw_lines = raw_text.splitlines()
    records = list(group_lines(raw_lines, args.group_lines))
    count = write_jsonl(records, output_path, args.dataset)

    print(f"Downloaded {len(raw_lines)} raw lines")
    print(f"Wrote {count} JSONL examples to {output_path}")
    if records:
        print("\nFirst example:")
        print("-" * 80)
        print(records[0][:300])
        print("-" * 80)


if __name__ == "__main__":
    main()
