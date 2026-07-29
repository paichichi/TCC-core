#!/usr/bin/env python3
"""Strictly convert a versioned TCC R3M checkpoint for RVT."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xirl.models import (  # pylint: disable=wrong-import-position
    HRAlignR3MBackbone,
    R3M_ADAPTER_INIT_TRAINABLE,
    R3M_LATE_ADAPTER_LAYOUT,
)


TRANSFER_FORMAT = "tcc_rvt_resnet_v1"
METHOD3_FORMAT = "asymmetric_domains_v2"
METHOD3_OBJECTIVES_V1 = "counterfactual_spatial_v1"


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file_obj:
    for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
  digest = hashlib.sha256()
  for key in sorted(state):
    value = state[key].detach().cpu().contiguous()
    digest.update(key.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def require_sha256(value, field_name: str) -> str:
  if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
    raise ValueError(f"{field_name} must be a lowercase SHA256 hex digest.")
  return value


def expected_r3m_state() -> dict[str, torch.Tensor]:
  model = HRAlignR3MBackbone(
      train_norm_affine=False,
      train_adapters=False,
      adapter_init=R3M_ADAPTER_INIT_TRAINABLE,
  )
  return model.state_dict()


def validate_state(
    state: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
  actual_keys = set(state)
  expected_keys = set(expected)
  missing = sorted(expected_keys - actual_keys)
  unexpected = sorted(actual_keys - expected_keys)
  if missing or unexpected:
    raise ValueError(
        "RVT backbone state is not exact: "
        f"missing={missing[:20]}, unexpected={unexpected[:20]}")

  mismatches = []
  dtype_mismatches = []
  nonfinite = []
  for key, value in state.items():
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"Checkpoint value is not a tensor: {key}")
    if value.shape != expected[key].shape:
      mismatches.append(
          (key, tuple(value.shape), tuple(expected[key].shape)))
    if value.dtype != expected[key].dtype:
      dtype_mismatches.append(
          (key, str(value.dtype), str(expected[key].dtype)))
    if torch.is_floating_point(value) and not torch.isfinite(value).all():
      nonfinite.append(key)
  if mismatches:
    raise ValueError(f"RVT backbone shape mismatches: {mismatches[:20]}")
  if dtype_mismatches:
    raise ValueError(
        f"RVT backbone dtype mismatches: {dtype_mismatches[:20]}")
  if nonfinite:
    raise ValueError(f"RVT backbone contains non-finite tensors: {nonfinite[:20]}")


def validate_source_format(
    checkpoint: dict,
    require_method3_v2: bool,
) -> dict:
  model_format = checkpoint.get("model_format", {})
  if not isinstance(model_format, dict):
    raise ValueError("Checkpoint model_format must be a dictionary.")
  layout = model_format.get("r3m_late_adapter_layout")
  if layout != R3M_LATE_ADAPTER_LAYOUT:
    raise ValueError(
        "Refusing a legacy or incompatible adapter topology: "
        f"expected {R3M_LATE_ADAPTER_LAYOUT!r}, got {layout!r}.")
  if require_method3_v2:
    required = {
        "adapter_domain": "robot_only",
        "representation_mode": "backbone_pooled",
        "r3m_adapter_init": R3M_ADAPTER_INIT_TRAINABLE,
    }
    mismatches = {
        key: (expected, model_format.get(key))
        for key, expected in required.items()
        if model_format.get(key) != expected
    }
    if mismatches:
      raise ValueError(
          f"Checkpoint is not a compatible Method 3 ResNet: {mismatches}")
    if model_format.get("method3_format") != METHOD3_FORMAT:
      raise ValueError(
          f"Unsupported Method 3 format: "
          f"expected {METHOD3_FORMAT!r}, "
          f"got {model_format.get('method3_format')!r}.")
    objective_format = model_format.get("method3_objectives")
    if objective_format is not None:
      if objective_format != METHOD3_OBJECTIVES_V1:
        raise ValueError(
            f"Unsupported Method 3 objectives: {objective_format!r}.")
      required_objectives = {
          "control_gain": "shared_teacher_hinge",
          "spatial_preservation":
              "same_robot_feature_map_relative_residual_hinge",
      }
      objective_mismatches = {
          key: (expected, model_format.get(key))
          for key, expected in required_objectives.items()
          if model_format.get(key) != expected
      }
      if objective_mismatches:
        raise ValueError(
            "Method 3 objective metadata is incomplete: "
            f"{objective_mismatches}")
  return model_format


def extract_backbone_state(checkpoint: dict) -> dict[str, torch.Tensor]:
  state = checkpoint.get("model", checkpoint)
  if not isinstance(state, dict):
    raise TypeError("Checkpoint model state must be a dictionary.")
  prefix = "backbone.convnet."
  converted = {
      key.removeprefix("backbone."): value
      for key, value in state.items()
      if key.startswith(prefix)
  }
  if not converted:
    raise ValueError("No backbone.convnet tensors found in checkpoint.")
  return converted


def convert(
    source: Path,
    target: Path,
    require_method3_v2: bool,
    force: bool = False,
) -> dict:
  if source.resolve() == target.resolve():
    raise ValueError("Source and target checkpoint paths must be different.")
  if target.exists() and not force:
    raise FileExistsError(
        f"Target already exists; validate it or pass --force: {target}")
  source_sha_before = file_sha256(source)
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  source_sha_after = file_sha256(source)
  if source_sha_before != source_sha_after:
    raise RuntimeError("Source checkpoint changed while it was being read.")
  if not isinstance(checkpoint, dict):
    raise TypeError("TCC checkpoint must be a dictionary.")
  source_format = validate_source_format(checkpoint, require_method3_v2)
  converted = extract_backbone_state(checkpoint)
  expected = expected_r3m_state()
  validate_state(converted, expected)

  source_sha = source_sha_before
  tensor_sha = tensor_state_sha256(converted)
  model_format = dict(source_format)
  model_format.update({
      "transfer_format": TRANSFER_FORMAT,
      "source_sha256": source_sha,
      "tensor_state_sha256": tensor_sha,
  })
  output = {
      "model": converted,
      "model_format": model_format,
      "source_checkpoint": str(source.resolve()),
      "source_step": checkpoint.get("step"),
  }

  target.parent.mkdir(parents=True, exist_ok=True)
  temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
  try:
    torch.save(output, temporary)
    os.replace(temporary, target)
  finally:
    temporary.unlink(missing_ok=True)
  return {
      "tensors": len(converted),
      "base_tensors": sum("late_adapter_" not in key for key in converted),
      "adapter_tensors": sum("late_adapter_" in key for key in converted),
      "source_sha256": source_sha,
      "tensor_state_sha256": tensor_sha,
  }


def validate_transfer(target: Path, source: Path | None = None) -> dict:
  checkpoint = torch.load(target, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, dict):
    raise TypeError("Transferred checkpoint must be a dictionary.")
  model_format = validate_source_format(
      checkpoint, require_method3_v2=True)
  if model_format.get("transfer_format") != TRANSFER_FORMAT:
    raise ValueError(
        f"Expected transfer_format={TRANSFER_FORMAT!r}, "
        f"got {model_format.get('transfer_format')!r}.")
  if model_format.get("r3m_late_adapter_layout") != R3M_LATE_ADAPTER_LAYOUT:
    raise ValueError("Transferred checkpoint has the wrong adapter layout.")

  state = checkpoint.get("model")
  if not isinstance(state, dict):
    raise TypeError("Transferred checkpoint has no model state.")
  validate_state(state, expected_r3m_state())
  tensor_sha = tensor_state_sha256(state)
  recorded_tensor_sha = require_sha256(
      model_format.get("tensor_state_sha256"), "tensor_state_sha256")
  if tensor_sha != recorded_tensor_sha:
    raise ValueError("Transferred checkpoint tensor digest does not match.")
  recorded_source_sha = require_sha256(
      model_format.get("source_sha256"), "source_sha256")
  if source is None:
    raise ValueError("Source checkpoint is required for transfer validation.")
  source_sha = file_sha256(source)
  if source_sha != recorded_source_sha:
    raise ValueError("Transferred checkpoint does not match its source.")
  return {
      "tensors": len(state),
      "base_tensors": sum("late_adapter_" not in key for key in state),
      "adapter_tensors": sum("late_adapter_" in key for key in state),
      "source_sha256": model_format.get("source_sha256"),
      "tensor_state_sha256": tensor_sha,
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

  convert_parser = subparsers.add_parser("convert")
  convert_parser.add_argument("source", type=Path)
  convert_parser.add_argument("target", type=Path)
  convert_parser.add_argument(
      "--force",
      action="store_true",
      help="Atomically replace an existing target after source validation.",
  )

  validate_parser = subparsers.add_parser("validate")
  validate_parser.add_argument("target", type=Path)
  validate_parser.add_argument("--source", type=Path, required=True)
  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.command == "convert":
    summary = convert(
        args.source,
        args.target,
        require_method3_v2=True,
        force=args.force,
    )
  else:
    summary = validate_transfer(args.target, args.source)
  print(
      f"{args.command} OK "
      f"tensors={summary['tensors']} "
      f"base={summary['base_tensors']} "
      f"adapter={summary['adapter_tensors']} "
      f"source_sha256={summary['source_sha256']} "
      f"tensor_sha256={summary['tensor_state_sha256']}")


if __name__ == "__main__":
  main()
