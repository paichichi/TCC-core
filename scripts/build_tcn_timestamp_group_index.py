#!/usr/bin/env python3
"""Build timestamp-aligned same-side multi-view groups for RH20T TCN training.

The output is a compact group index. Each row is one target timestamp for one
episode/side, and ``views_json`` contains all views whose nearest frame is within
the timestamp threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcc-root", type=Path, default=Path("/home/paichichi/data/RH20T/TCC_RH20T"))
    parser.add_argument("--rgb-root", type=Path, default=Path("/mnt/g/RH20T/RGB"))
    parser.add_argument(
        "--cleaned-json",
        type=Path,
        default=Path("/home/paichichi/data/RH20T/rh20t_cleaned_data.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--threshold-ms", type=int, default=33)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--respect-start-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roles", nargs="+", default=["h", "r"], choices=["h", "r"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cleaned_data(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {key.replace("RH100T_", "RH20T_"): value for key, value in raw.items()}


def nearest_indices(video_timestamps: np.ndarray, cleaned_timestamps: np.ndarray):
    video_ts = np.asarray(video_timestamps, dtype=np.int64)
    cleaned_ts = np.asarray(cleaned_timestamps, dtype=np.int64)
    idxs = np.searchsorted(video_ts, cleaned_ts)
    idxs = np.clip(idxs, 0, len(video_ts) - 1)
    prev = np.clip(idxs - 1, 0, len(video_ts) - 1)
    choose_prev = np.abs(video_ts[prev] - cleaned_ts) < np.abs(video_ts[idxs] - cleaned_ts)
    idxs[choose_prev] = prev[choose_prev]
    matched_ts = video_ts[idxs]

    dedup_idxs = []
    dedup_matched = []
    seen = set()
    for idx, mts in zip(idxs.tolist(), matched_ts.tolist()):
        if idx in seen:
            continue
        seen.add(idx)
        dedup_idxs.append(idx)
        dedup_matched.append(mts)
    return np.asarray(dedup_idxs, dtype=np.int64), np.asarray(dedup_matched, dtype=np.int64)


def raw_timestamps_path(rgb_root: Path, cfg: str, episode: str, camera_view: str) -> Path:
    return rgb_root / cfg / episode / camera_view / "color" / "timestamps.npy"


def load_sequence_timestamps(
    row: dict[str, str],
    role: str,
    rgb_root: Path,
    cleaned_data: dict[str, Any],
) -> np.ndarray:
    cfg = row["cfg"]
    camera_view = row["camera_view"]
    if role == "h":
        ts_path = raw_timestamps_path(rgb_root, cfg, row["human_episode_id"], camera_view)
        return np.load(ts_path).astype(np.int64)

    ts_path = raw_timestamps_path(rgb_root, cfg, row["robot_episode_id"], camera_view)
    video_ts = np.load(ts_path).astype(np.int64)
    cleaned_ts = np.asarray(
        cleaned_data[cfg][row["robot_episode_id"]][camera_view],
        dtype=np.int64,
    )
    _, matched_ts = nearest_indices(video_ts, cleaned_ts)
    return matched_ts


def nearest_index(timestamps: np.ndarray, target_ts: int) -> int:
    pos = int(np.searchsorted(timestamps, target_ts))
    candidates = []
    if pos < len(timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    return min(candidates, key=lambda idx: abs(int(timestamps[idx]) - target_ts))


def frame_rel_path(class_name: str, sequence_id: str, frame_idx: int) -> str:
    return f"train/{class_name}/{sequence_id}/{frame_idx:06d}.jpg"


def sequence_record(
    lookup_row: dict[str, str],
    manifest_by_seq: dict[str, dict[str, str]],
    role: str,
    rgb_root: Path,
    cleaned_data: dict[str, Any],
    respect_start_frame: bool,
) -> dict[str, Any] | None:
    sequence_id = lookup_row["human_sequence_id"] if role == "h" else lookup_row["robot_sequence_id"]
    manifest = manifest_by_seq[sequence_id]
    num_frames = int(manifest["num_frames"])
    start_frame = int(manifest.get("start_frame", 0) or 0) if respect_start_frame else 0
    timestamps = load_sequence_timestamps(lookup_row, role, rgb_root, cleaned_data)
    usable = min(num_frames, len(timestamps))
    if usable <= start_frame:
        return None
    timestamps = timestamps[:usable]
    valid_frames = np.arange(start_frame, usable, dtype=np.int64)
    valid_timestamps = timestamps[start_frame:usable]
    return {
        "sequence_id": sequence_id,
        "camera_id": manifest["camera_id"],
        "camera_view": lookup_row["camera_view"],
        "num_frames": usable,
        "start_frame": start_frame,
        "timestamps": timestamps,
        "valid_frames": valid_frames,
        "valid_timestamps": valid_timestamps,
    }


def build_group_rows_for_side(
    lookup_rows: list[dict[str, str]],
    manifest_by_seq: dict[str, dict[str, str]],
    role: str,
    rgb_root: Path,
    cleaned_data: dict[str, Any],
    threshold_ms: int,
    min_views: int,
    respect_start_frame: bool,
) -> list[dict[str, Any]]:
    sequences = []
    for row in lookup_rows:
        rec = sequence_record(row, manifest_by_seq, role, rgb_root, cleaned_data, respect_start_frame)
        if rec is not None:
            sequences.append(rec)
    if len(sequences) < min_views:
        return []

    ref = max(sequences, key=lambda rec: len(rec["valid_timestamps"]))
    group_rows = []
    base = lookup_rows[0]
    class_name = base["class_name"]
    raw_episode = base["human_episode_id"] if role == "h" else base["robot_episode_id"]
    for target_ts in ref["valid_timestamps"].tolist():
        views = []
        max_abs_delta = 0
        for seq in sequences:
            idx = nearest_index(seq["timestamps"], int(target_ts))
            if idx < seq["start_frame"] or idx >= seq["num_frames"]:
                continue
            ts = int(seq["timestamps"][idx])
            delta = ts - int(target_ts)
            abs_delta = abs(delta)
            if abs_delta > threshold_ms:
                continue
            max_abs_delta = max(max_abs_delta, abs_delta)
            views.append(
                {
                    "sequence_id": seq["sequence_id"],
                    "camera_id": int(seq["camera_id"]),
                    "camera_view": seq["camera_view"],
                    "frame_idx": int(idx),
                    "timestamp_ms": ts,
                    "delta_ms": int(delta),
                    "rel_path": frame_rel_path(class_name, seq["sequence_id"], int(idx)),
                }
            )
        if len(views) < min_views:
            continue
        group_rows.append(
            {
                "episode_id": base["episode_id"],
                "class_name": class_name,
                "task_id": base["task_id"],
                "role": role,
                "cfg": base["cfg"],
                "raw_episode_id": raw_episode,
                "target_timestamp_ms": int(target_ts),
                "ref_sequence_id": ref["sequence_id"],
                "ref_camera_id": int(ref["camera_id"]),
                "ref_camera_view": ref["camera_view"],
                "num_views": len(views),
                "max_abs_delta_ms": max_abs_delta,
                "views_json": json.dumps(views, separators=(",", ":"), ensure_ascii=False),
            }
        )
    return group_rows


def main() -> None:
    args = parse_args()
    output = args.output or (args.tcc_root / "tcn_timestamp_groups.csv")
    summary_output = args.summary_output or (args.tcc_root / "tcn_timestamp_groups_summary.json")

    manifest_rows = read_csv(args.tcc_root / "manifest.csv")
    lookup_rows = read_csv(args.tcc_root / "lookup.csv")
    manifest_by_seq = {row["sequence_id"]: row for row in manifest_rows}

    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in lookup_rows:
        by_episode[row["episode_id"]].append(row)
    episode_ids = sorted(by_episode, key=lambda value: int(value))
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    cleaned_data = load_cleaned_data(args.cleaned_json) if "r" in args.roles else {}

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_id",
        "episode_id",
        "class_name",
        "task_id",
        "role",
        "cfg",
        "raw_episode_id",
        "target_timestamp_ms",
        "ref_sequence_id",
        "ref_camera_id",
        "ref_camera_view",
        "num_views",
        "max_abs_delta_ms",
        "views_json",
    ]

    summary = {
        "threshold_ms": args.threshold_ms,
        "min_views": args.min_views,
        "respect_start_frame": args.respect_start_frame,
        "episodes_processed": 0,
        "groups": 0,
        "by_role": {role: {"episodes": 0, "groups": 0, "view_refs": 0} for role in args.roles},
    }

    group_id = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, episode_id in enumerate(episode_ids, start=1):
            rows = by_episode[episode_id]
            summary["episodes_processed"] += 1
            for role in args.roles:
                group_rows = build_group_rows_for_side(
                    rows,
                    manifest_by_seq,
                    role,
                    args.rgb_root,
                    cleaned_data,
                    args.threshold_ms,
                    args.min_views,
                    args.respect_start_frame,
                )
                if group_rows:
                    summary["by_role"][role]["episodes"] += 1
                for row in group_rows:
                    row["group_id"] = group_id
                    writer.writerow(row)
                    group_id += 1
                    summary["groups"] += 1
                    summary["by_role"][role]["groups"] += 1
                    summary["by_role"][role]["view_refs"] += int(row["num_views"])
            if args.progress_every and idx % args.progress_every == 0:
                print(f"processed {idx} / {len(episode_ids)} episodes, groups={group_id}", flush=True)

    for role, role_summary in summary["by_role"].items():
        groups = role_summary["groups"]
        role_summary["mean_views_per_group"] = role_summary["view_refs"] / groups if groups else 0.0

    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote: {output}")
    print(f"wrote: {summary_output}")


if __name__ == "__main__":
    main()
