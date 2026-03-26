#!/usr/bin/env python3
"""
Validate a standalone submission inference script.

This is intended for both:
- students, to self-check their packaged submission
- instructors/TAs, to quickly verify the common interface
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_KEYS = [
    "submission_name",
    "prompt",
    "generated_text",
    "response_text",
    "num_generated_tokens",
    "wall_time_sec",
    "seconds_per_generated_token",
    "tokens_per_second",
    "parameter_count",
    "artifact_size_bytes",
    "device",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the standalone Moodle inference interface")
    parser.add_argument("--script", type=str, required=True, help="Path to standalone_inference.py")
    parser.add_argument("--prompt", type=str, default="Once upon a time,", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(args.script).resolve()

    cmd = [
        sys.executable,
        str(script_path),
        "--prompt",
        args.prompt,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--top_p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]

    proc = subprocess.run(
        cmd,
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        print("Validation failed: script exited with non-zero status.", file=sys.stderr)
        if proc.stdout:
            print("\n--- stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("\n--- stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print("Validation failed: stdout is not valid JSON.", file=sys.stderr)
        print(f"JSON error: {exc}", file=sys.stderr)
        if proc.stdout:
            print("\n--- stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print("\n--- stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
        return 1

    missing = [key for key in REQUIRED_KEYS if key not in result]
    if missing:
        print("Validation failed: missing required keys.", file=sys.stderr)
        print(", ".join(missing), file=sys.stderr)
        return 1

    print("Standalone submission interface: OK")
    print(json.dumps(
        {
            "submission_name": result["submission_name"],
            "device": result["device"],
            "parameter_count": result["parameter_count"],
            "artifact_size_bytes": result["artifact_size_bytes"],
            "num_generated_tokens": result["num_generated_tokens"],
            "wall_time_sec": result["wall_time_sec"],
            "tokens_per_second": result["tokens_per_second"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
