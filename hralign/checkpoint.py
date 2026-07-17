from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from .models import HRAlignR3ML, load_torch_checkpoint


def unwrap_model(model: torch.nn.Module) -> HRAlignR3ML:
    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, HRAlignR3ML):
        raise TypeError(f"Expected HRAlignR3ML, got {type(candidate).__name__}.")
    return candidate


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    epoch: int,
    config: Mapping[str, Any],
) -> None:
    core = unwrap_model(model)
    payload = {
        "format": "hralign-reproduction-v1",
        "step": step,
        "epoch": epoch,
        "trainable_state": core.trainable_state(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "config": dict(config),
    }
    _atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int]:
    checkpoint = load_torch_checkpoint(path)
    if checkpoint.get("format") != "hralign-reproduction-v1":
        raise ValueError(f"Not an HR-Align training checkpoint: {path}")
    core = unwrap_model(model)
    core.load_trainable_state(checkpoint["trainable_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return int(checkpoint["step"]), int(checkpoint.get("epoch", 0))


def export_official_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    steps_per_epoch: int,
    config: Mapping[str, Any],
) -> None:
    """Exports the model_state layout consumed by official HumanRobotAlign."""
    core = unwrap_model(model)
    epoch = max(0, math.ceil(step / max(steps_per_epoch, 1)) - 1)
    clean_config = {key: value for key, value in config.items() if key != "_meta"}
    payload = {
        "epoch": epoch,
        "model_state": core.official_model_state(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": yaml.safe_dump(clean_config, sort_keys=False),
        "reproduction": {
            "format": "hralign-reproduction-v1",
            "step": step,
            "paper_target": "R3M-Align-L",
        },
    }
    _atomic_torch_save(payload, path)


def compare_model_state_layout(
    produced: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    produced_keys = set(produced)
    reference_keys = set(reference)
    shared = produced_keys & reference_keys
    shape_mismatches = {
        key: (tuple(produced[key].shape), tuple(reference[key].shape))
        for key in sorted(shared)
        if produced[key].shape != reference[key].shape
    }
    return {
        "missing": sorted(reference_keys - produced_keys),
        "unexpected": sorted(produced_keys - reference_keys),
        "shape_mismatches": shape_mismatches,
        "shared": len(shared),
    }
