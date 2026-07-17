#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hralign.text import FrozenR3MTextEncoder, load_r3m_language_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache one deterministic DistilBERT embedding per RH20T task."
    )
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--task-descriptions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer", default="distilbert-base-uncased"
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptions = json.loads(
        Path(args.task_descriptions).read_text(encoding="utf-8")
    )
    language_state = load_r3m_language_state(args.pretrain)
    encoder = FrozenR3MTextEncoder(
        language_state,
        tokenizer_name=args.tokenizer,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        cache_by_text=False,
    ).to(args.device)

    embeddings = {}
    for task_id, value in descriptions.items():
        text = (
            value
            if isinstance(value, str)
            else value["task_description_english"]
        )
        embeddings[task_id] = encoder([text])[0].cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "r3m-distilbert-single-text-mean-v1",
            "embeddings": embeddings,
            "note": (
                "Fast mode: each text was encoded without batch padding. "
                "This is deterministic but not bit-identical to R3M's "
                "batch-padding-dependent mean."
            ),
        },
        output,
    )
    print(f"saved {len(embeddings)} task embeddings to {output}")


if __name__ == "__main__":
    main()
