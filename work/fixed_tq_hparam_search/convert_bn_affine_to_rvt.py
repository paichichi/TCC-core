#!/usr/bin/env python3
"""Strictly convert a BN-affine R3M TCC checkpoint for RVT2."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


TRANSFER_FORMAT = "tcc_rvt_r3m_bn_affine_v1"
REQUIRED_ARGS = {
    "backbone": "r3m_resnet50",
    "train_backbone_adapters": False,
    "train_backbone_norm_affine": True,
    "adapter_domain": "all",
    "aux_teacher_stop_grad": False,
    "aux_global_negatives": False,
    "view_mode": "single",
    "representation_mode": "backbone_pooled",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def argument_value(arguments: object, key: str) -> object:
    if isinstance(arguments, dict):
        return arguments.get(key)
    return getattr(arguments, key, None)


def validate_source_arguments(checkpoint: dict) -> None:
    arguments = checkpoint.get("args")
    if arguments is None:
        raise ValueError("Source checkpoint has no saved training arguments.")
    mismatches = {
        key: {"expected": expected, "actual": argument_value(arguments, key)}
        for key, expected in REQUIRED_ARGS.items()
        if argument_value(arguments, key) != expected
    }
    if mismatches:
        raise ValueError(f"Source is not the required symmetric BN-affine design: {mismatches}")


def extract_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    source_state = checkpoint.get("model", checkpoint)
    if not isinstance(source_state, dict):
        raise TypeError("Source model state is not a dictionary.")
    state = {
        key.removeprefix("backbone."): value
        for key, value in source_state.items()
        if key.startswith("backbone.convnet.")
    }
    if not state:
        raise ValueError("No backbone.convnet tensors found.")
    return state


def reference_state(reference_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("r3m"), dict):
        raise ValueError("Reference must contain the original R3M 'r3m' state.")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint["r3m"].items()
        if key.startswith("module.convnet.")
    }


def validate_state(
    state: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> None:
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    if missing or unexpected:
        raise ValueError(
            f"Converted state key mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    shape_mismatches = [
        (key, tuple(value.shape), tuple(reference[key].shape))
        for key, value in state.items()
        if value.shape != reference[key].shape
    ]
    if shape_mismatches:
        raise ValueError(f"Converted state shape mismatch: {shape_mismatches[:10]}")
    nonfinite = [
        key
        for key, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f"Converted state contains non-finite tensors: {nonfinite[:10]}")


def convert(source: Path, target: Path, reference_path: Path, force: bool) -> None:
    if target.exists() and not force:
        raise FileExistsError(f"Target exists; validate it or pass --force: {target}")
    source_hash_before = file_sha256(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_hash_after = file_sha256(source)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source changed while being read.")
    if not isinstance(checkpoint, dict):
        raise TypeError("Source checkpoint is not a dictionary.")

    validate_source_arguments(checkpoint)
    state = extract_state(checkpoint)
    validate_state(state, reference_state(reference_path))
    tensor_hash = tensor_state_sha256(state)
    output = {
        "model": state,
        "source_checkpoint": str(source.resolve()),
        "source_step": checkpoint.get("step"),
        "model_format": {
            "transfer_format": TRANSFER_FORMAT,
            "finetuning": "bn_affine_only",
            "infonce_direction": "symmetric",
            "source_sha256": source_hash_before,
            "tensor_state_sha256": tensor_hash,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(output, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def validate(target: Path, source: Path, reference_path: Path) -> None:
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Converted checkpoint is not a dictionary.")
    metadata = checkpoint.get("model_format")
    if not isinstance(metadata, dict) or metadata.get("transfer_format") != TRANSFER_FORMAT:
        raise ValueError("Converted checkpoint has invalid transfer metadata.")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise TypeError("Converted checkpoint has no model state.")
    validate_state(state, reference_state(reference_path))
    if metadata.get("source_sha256") != file_sha256(source):
        raise ValueError("Converted checkpoint does not match the source file.")
    if metadata.get("tensor_state_sha256") != tensor_state_sha256(state):
        raise ValueError("Converted checkpoint tensor hash mismatch.")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("target", type=Path)
    convert_parser.add_argument("--reference", required=True, type=Path)
    convert_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("target", type=Path)
    validate_parser.add_argument("--source", required=True, type=Path)
    validate_parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "convert":
        convert(args.source, args.target, args.reference, args.force)
    else:
        validate(args.target, args.source, args.reference)


if __name__ == "__main__":
    main()
