#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from hralign.models import extract_model_state, load_torch_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit released AdaptedR3M against original R3M."
    )
    parser.add_argument("--unadapted", required=True)
    parser.add_argument("--adapted", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unadapted_checkpoint = load_torch_checkpoint(args.unadapted)
    adapted_checkpoint = load_torch_checkpoint(args.adapted)
    unadapted = {
        key.removeprefix("module."): value
        for key, value in extract_model_state(unadapted_checkpoint).items()
    }
    adapted = extract_model_state(adapted_checkpoint)

    changed = []
    for key, value in adapted.items():
        if key not in unadapted or value.shape != unadapted[key].shape:
            continue
        difference = (value.float() - unadapted[key].float()).abs()
        if difference.max().item() != 0:
            changed.append(key)

    adapter_tensors = {
        key: value for key, value in adapted.items() if "late_adapter_" in key
    }
    adapter_parameters = sum(value.numel() for value in adapter_tensors.values())
    language_projection = sum(
        value.numel()
        for key, value in adapted.items()
        if key.startswith("lang_linear.")
    )
    changed_types = Counter(
        "BN running buffer"
        if any(
            suffix in key
            for suffix in ("running_mean", "running_var", "num_batches_tracked")
        )
        else "parameter"
        for key in changed
    )

    print(f"adapted_epoch: {adapted_checkpoint.get('epoch')}")
    optimizer = adapted_checkpoint.get("optimizer_state", {})
    states = optimizer.get("state", {})
    steps = sorted(
        {
            int(state["step"])
            for state in states.values()
            if isinstance(state, dict) and "step" in state
        }
    )
    print(f"optimizer_steps: {steps}")
    learning_rates = [
        group.get("lr") for group in optimizer.get("param_groups", [])
    ]
    print(f"optimizer_learning_rates: {learning_rates}")
    released_epoch = adapted_checkpoint.get("epoch")
    steps_per_epoch = None
    if (
        isinstance(released_epoch, int)
        and len(steps) == 1
        and steps[0] % (released_epoch + 1) == 0
    ):
        steps_per_epoch = steps[0] // (released_epoch + 1)
        completed_steps = steps[0] - 1
        epoch_exact = completed_steps / steps_per_epoch
        warmup_lr = 1e-6 + epoch_exact / 10.0 * (1e-4 - 1e-6)
        print(f"inferred_steps_per_epoch: {steps_per_epoch}")
        print(f"inferred_epoch_exact: {epoch_exact:.15f}")
        print(f"slowfast_expected_lr: {warmup_lr:.15g}")
        if learning_rates and not math.isclose(
            learning_rates[0], warmup_lr, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(
                "Released optimizer LR does not match the inferred "
                "SlowFast warmup schedule."
            )
    batch_counter_deltas = [
        int(value) - int(unadapted[key])
        for key, value in adapted.items()
        if key.endswith("num_batches_tracked") and key in unadapted
    ]
    counter_delta_counts = Counter(batch_counter_deltas)
    print(f"bn_counter_modules: {len(batch_counter_deltas)}")
    print(f"bn_counter_delta_counts: {dict(counter_delta_counts)}")
    if (
        len(counter_delta_counts) == 1
        and len(steps) == 1
        and steps_per_epoch is not None
    ):
        total_delta = next(iter(counter_delta_counts))
        alignment_three_stream_delta = steps[0] * 3
        predecessor_delta = total_delta - alignment_three_stream_delta
        print(
            "bn_counter_three_stream_alignment_delta: "
            f"{alignment_three_stream_delta}"
        )
        print(f"bn_counter_inferred_predecessor_delta: {predecessor_delta}")
        print(
            "bn_counter_inferred_predecessor_epochs: "
            f"{predecessor_delta / steps_per_epoch:.6g}"
        )
    zero_first_moment = [
        parameter_id
        for parameter_id, state in states.items()
        if isinstance(state, dict)
        and isinstance(state.get("exp_avg"), torch.Tensor)
        and torch.count_nonzero(state["exp_avg"]).item() == 0
    ]
    print(f"optimizer_zero_first_moment_ids: {zero_first_moment}")
    print(f"model_tensors: {len(adapted)}")
    print(f"adapter_tensors: {len(adapter_tensors)}")
    print(f"adapter_parameters: {adapter_parameters}")
    print(f"language_projection_parameters: {language_projection}")
    print(f"shared_changed: {dict(changed_types)}")
    print(
        "changed_non_buffers:",
        [
            key
            for key in changed
            if not any(
                suffix in key
                for suffix in (
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                )
            )
        ],
    )

    zero_mapping = [
        key
        for key, value in adapter_tensors.items()
        if ".D_mapping." in key and torch.count_nonzero(value).item() == 0
    ]
    print(f"zero_D_mapping_tensors: {len(zero_mapping)}/6")

    raw_config = adapted_checkpoint.get("cfg")
    if isinstance(raw_config, str):
        config = yaml.safe_load(raw_config)
        print("released_training_config:")
        for section, key in (
            ("MODEL", "ADAPTER"),
            ("MODEL", "FROZEN_BN"),
            ("MODEL", "LOSS_TEMPORATURE"),
            ("DATA", "NUM_FRAMES"),
            ("DATA", "SAMPLING_RATE"),
            ("DATA", "TARGET_FPS"),
            ("DATA", "TRAIN_CROP_SIZE"),
            ("DATA", "TRAIN_JITTER_SCALES"),
            ("DATA", "RANDOM_FLIP"),
            ("TRAIN", "BATCH_SIZE"),
            ("TRAIN", "CHECKPOINT_FILE_PATH"),
            ("TRAIN", "LOAD_CHECKPOINT_MANUAL"),
            ("SOLVER", "BASE_LR"),
            ("SOLVER", "COSINE_END_LR"),
            ("SOLVER", "OPTIMIZING_METHOD"),
            ("SOLVER", "WEIGHT_DECAY"),
            ("SOLVER", "WARMUP_EPOCHS"),
            ("SOLVER", "WARMUP_START_LR"),
            ("SOLVER", "MAX_EPOCH"),
            ("AUG", "ENABLE"),
            ("OUTPUT_DIR", None),
            ("NUM_GPUS", None),
        ):
            value = config.get(section)
            if key is not None and isinstance(value, dict):
                value = value.get(key)
            print(f"  {section}{'.' + key if key else ''}: {value}")


if __name__ == "__main__":
    main()
