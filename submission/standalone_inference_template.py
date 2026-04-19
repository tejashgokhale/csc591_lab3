#!/usr/bin/env python3
"""
Standalone inference template for Moodle submission.

Contract:
1. This script must run from inside the packaged submission folder.
2. It must accept the standard CLI arguments below.
3. It must print exactly one JSON object to stdout.

Students may change the model architecture however they want, as long as this
script still loads the packaged weights and obeys the output schema.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PACKAGE_DIR = Path(__file__).resolve().parent


def maybe_set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: Any) -> int | None:
    if model is None or not hasattr(model, "parameters"):
        return None
    return int(sum(p.numel() for p in model.parameters()))


def total_artifact_size_bytes(paths: list[str | Path]) -> int:
    total = 0
    for path in paths:
        resolved = (PACKAGE_DIR / path).resolve() if not Path(path).is_absolute() else Path(path)
        if resolved.exists() and resolved.is_file():
            total += resolved.stat().st_size
    return total



def load_runtime(device: str) -> dict[str, Any]:
    from src.model.config import ModelConfig
    from src.model.language_model import TransformerLanguageModel
    from src.tokenizer.loading import load_tokenizer

    # paths inside submission folder
    checkpoint_path = PACKAGE_DIR / "best_model.pt"
    tokenizer_path = PACKAGE_DIR / "english_bytebpe_8k.json"

    # load tokenizer
    tokenizer = load_tokenizer(str(tokenizer_path))

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # rebuild model config
    model_config_dict = checkpoint["config"]["model_config"]
    model_config = ModelConfig.from_dict(model_config_dict)

    # create model
    model = TransformerLanguageModel(model_config)

    # fix DataParallel prefix if needed
    state_dict = checkpoint["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "submission_name": "tiny_transformer",
        "dtype": "float32",
        "artifact_paths": [
            "best_model.pt",
            "english_bytebpe_8k.json",
        ],
    }



def generate_with_runtime(
    runtime: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> dict[str, Any]:

    model = runtime["model"]
    tokenizer = runtime["tokenizer"]

    device = next(model.parameters()).device

    # encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            do_sample=(temperature != 1.0 or top_k > 0),
        )

    # decode
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    # extract only new tokens
    prompt_len = input_ids.shape[1]
    response_ids = output_ids[0][prompt_len:]
    response_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "response_text": response_text,
        "num_generated_tokens": int(len(response_ids)),
        "dtype": "float32",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone inference entry point for Moodle submission")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of new tokens")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0 disables)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to also save the JSON result",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maybe_set_seed(args.seed)
    device = resolve_device(args.device)

    runtime = load_runtime(device=device)

    start_time = time.perf_counter()
    generation = generate_with_runtime(
        runtime=runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    wall_time_sec = time.perf_counter() - start_time

    num_generated_tokens = int(generation["num_generated_tokens"])
    seconds_per_generated_token = (
        wall_time_sec / num_generated_tokens if num_generated_tokens > 0 else None
    )
    tokens_per_second = (
        num_generated_tokens / wall_time_sec if wall_time_sec > 0 and num_generated_tokens > 0 else None
    )

    artifact_paths = generation.get("artifact_paths") or runtime.get("artifact_paths") or []
    parameter_count = generation.get("parameter_count")
    if parameter_count is None:
        parameter_count = count_parameters(runtime.get("model"))

    result = {
        "submission_name": runtime.get("submission_name", PACKAGE_DIR.name),
        "prompt": args.prompt,
        "generated_text": generation["generated_text"],
        "response_text": generation["response_text"],
        "num_generated_tokens": num_generated_tokens,
        "wall_time_sec": wall_time_sec,
        "seconds_per_generated_token": seconds_per_generated_token,
        "tokens_per_second": tokens_per_second,
        "parameter_count": parameter_count,
        "artifact_size_bytes": total_artifact_size_bytes(artifact_paths),
        "device": device,
        "dtype": generation.get("dtype", runtime.get("dtype")),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "extra": generation.get("extra", {}),
    }

    json_text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + "\n", encoding="utf-8")

    sys.stdout.write(json_text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
