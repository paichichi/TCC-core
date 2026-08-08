#!/usr/bin/env python3
"""Convert a completed BN-affine TCC checkpoint into RVT ResNet pretrain format."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    source_hash = sha256(args.source)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != 40000:
        raise ValueError(f"Expected source step 40000, got {checkpoint.get('step')}")

    source_state = checkpoint.get("model", checkpoint)
    state = {
        key.removeprefix("backbone."): value
        for key, value in source_state.items()
        if key.startswith("backbone.convnet.")
    }
    if len(state) != 318:
        raise ValueError(f"Expected 318 ResNet tensors, got {len(state)}")
    nonfinite = [
        key
        for key, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f"Non-finite tensors: {nonfinite[:20]}")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.target.with_name(f".{args.target.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "model": state,
            "source_checkpoint": str(args.source),
            "source_step": 40000,
            "source_sha256": source_hash,
            "tensors": len(state),
            "finetuning": "bn_affine_only",
            "infonce_direction": "symmetric",
        },
        temporary,
    )
    os.replace(temporary, args.target)
    print(
        f"converted tensors={len(state)} source_sha256={source_hash} "
        f"target_sha256={sha256(args.target)}"
    )


if __name__ == "__main__":
    main()
