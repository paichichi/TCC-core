from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot override {dotted_key!r}: {key!r} is not a mapping."
            )
        target = child
    target[keys[-1]] = value


def apply_overrides(
    config: dict[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override {override!r} must use dotted.path=value syntax."
            )
        key, raw_value = override.split("=", 1)
        _set_nested(result, key, yaml.safe_load(raw_value))
    return result


def _expand_path(value: str, project_root: Path) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = project_root / expanded
    return str(expanded.resolve())


def _resolve_paths(config: dict[str, Any], project_root: Path) -> None:
    path_keys = {
        ("data", "root"),
        ("data", "lookup"),
        ("data", "manifest"),
        ("data", "task_descriptions"),
        ("model", "pretrain"),
        ("text", "cache_dir"),
        ("text", "embeddings"),
        ("train", "output_dir"),
    }
    for section, key in path_keys:
        value = config.get(section, {}).get(key)
        if isinstance(value, str) and value:
            config[section][key] = _expand_path(value, project_root)


def load_config(
    path: str | Path, overrides: Iterable[str] = ()
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")

    config = apply_overrides(config, overrides)
    project_root = config_path.parent.parent
    _resolve_paths(config, project_root)
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(project_root),
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = [
        ("data", "root"),
        ("data", "lookup"),
        ("data", "manifest"),
        ("data", "task_descriptions"),
        ("model", "pretrain"),
        ("train", "output_dir"),
    ]
    missing = [
        f"{section}.{key}"
        for section, key in required
        if not config.get(section, {}).get(key)
    ]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")

    num_frames = int(config.get("sampling", {}).get("num_frames", 5))
    crop_size = int(config.get("sampling", {}).get("crop_size", 224))
    frame_index_mode = str(
        config.get("sampling", {}).get("frame_index_mode", "compact")
    )
    sampling_config = config.get("sampling", {})
    sampling_mode = str(sampling_config.get("mode", "slowfast_clip"))
    spatial_mode = str(
        sampling_config.get("spatial_mode", "random_resized_crop")
    )
    relative_crop_scale = sampling_config.get(
        "relative_crop_scale", [0.08, 1.0]
    )
    relative_crop_aspect = sampling_config.get(
        "relative_crop_aspect", [0.75, 1.3333]
    )
    adapted_bn_mode = str(
        config.get("model", {}).get(
            "adapted_bn_mode", "shared_stream_stats"
        )
    )
    batch_size = int(config.get("train", {}).get("batch_size_per_gpu", 1))
    max_pairs = config.get("data", {}).get("max_pairs")
    temperature = float(config.get("loss", {}).get("temperature", 0.1))
    max_steps = int(config.get("train", {}).get("max_steps", 8000))
    train_config = config.get("train", {})
    warmup_epochs = float(train_config.get("warmup_epochs", 10.0))
    schedule_epochs = float(train_config.get("schedule_epochs", 300.0))
    deprecated_schedule_keys = [
        key
        for key in ("warmup_steps", "schedule_total_steps")
        if key in train_config
    ]
    if num_frames < 1:
        raise ValueError("sampling.num_frames must be positive.")
    if crop_size < 1:
        raise ValueError("sampling.crop_size must be positive.")
    if frame_index_mode not in {"compact", "manifest_offset"}:
        raise ValueError(
            "sampling.frame_index_mode must be compact or manifest_offset."
        )
    if sampling_mode not in {"slowfast_clip", "random_sorted"}:
        raise ValueError(
            "sampling.mode must be slowfast_clip or random_sorted."
        )
    if spatial_mode not in {
        "random_resized_crop",
        "short_side_jitter",
    }:
        raise ValueError(
            "sampling.spatial_mode must be random_resized_crop or "
            "short_side_jitter."
        )
    for name, values, require_positive in (
        ("relative_crop_scale", relative_crop_scale, True),
        ("relative_crop_aspect", relative_crop_aspect, True),
    ):
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 2
            or (require_positive and float(values[0]) <= 0)
            or float(values[0]) > float(values[1])
        ):
            raise ValueError(
                f"sampling.{name} must be an increasing positive pair."
            )
    if adapted_bn_mode not in {
        "shared_stream_stats",
        "robot_stats",
        "frozen",
    }:
        raise ValueError(
            "model.adapted_bn_mode must be shared_stream_stats, "
            "robot_stats, or frozen."
        )
    if batch_size < 1:
        raise ValueError("train.batch_size_per_gpu must be positive.")
    if max_pairs is not None and (
        isinstance(max_pairs, bool)
        or not isinstance(max_pairs, int)
        or max_pairs < 1
    ):
        raise ValueError("data.max_pairs must be null or a positive integer.")
    if temperature <= 0:
        raise ValueError("loss.temperature must be positive.")
    if max_steps < 1:
        raise ValueError("train.max_steps must be positive.")
    if deprecated_schedule_keys:
        raise ValueError(
            "Use epoch-based train.warmup_epochs and train.schedule_epochs; "
            "remove deprecated keys: "
            + ", ".join(deprecated_schedule_keys)
        )
    if warmup_epochs < 0:
        raise ValueError("train.warmup_epochs cannot be negative.")
    if schedule_epochs <= warmup_epochs:
        raise ValueError(
            "train.schedule_epochs must be greater than train.warmup_epochs."
        )


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if key != "_meta"}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)
