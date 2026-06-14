#!/usr/bin/env python3
"""Full-dataset same-side multi-view timestamp alignment statistics."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, default=Path("/mnt/g/RH20T/RGB"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/same_side_timestamp_alignment_stats"),
    )
    parser.add_argument("--thresholds-ms", type=float, nargs="+", default=[16.0, 20.0, 33.0])
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_timestamps(episode_dir: Path) -> dict[str, np.ndarray]:
    cameras: dict[str, np.ndarray] = {}
    for cam_dir in sorted(episode_dir.glob("cam_*")):
        ts_path = cam_dir / "color" / "timestamps.npy"
        if not ts_path.exists():
            continue
        try:
            ts = np.load(ts_path).astype(np.int64)
        except Exception:
            continue
        if len(ts) > 0:
            cameras[cam_dir.name] = ts
    return cameras


def nearest_abs_deltas(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """For each source timestamp, return nearest absolute delta to target timestamps."""
    pos = np.searchsorted(target_ts, source_ts)
    deltas = np.full(len(source_ts), np.iinfo(np.int64).max, dtype=np.int64)
    valid_right = pos < len(target_ts)
    if np.any(valid_right):
        deltas[valid_right] = np.abs(target_ts[pos[valid_right]] - source_ts[valid_right])
    valid_left = pos > 0
    if np.any(valid_left):
        left = np.abs(target_ts[pos[valid_left] - 1] - source_ts[valid_left])
        deltas[valid_left] = np.minimum(deltas[valid_left], left)
    return deltas


def common_overlap(cameras: dict[str, np.ndarray]) -> tuple[int, int] | None:
    if len(cameras) < 2:
        return None
    start = max(int(ts[0]) for ts in cameras.values())
    end = min(int(ts[-1]) for ts in cameras.values())
    if end <= start:
        return None
    return start, end


def count_all_view_groups(
    cameras: dict[str, np.ndarray],
    thresholds: list[float],
) -> tuple[dict[str, int], int, int | None]:
    overlap = common_overlap(cameras)
    if overlap is None:
        return {str(t): 0 for t in thresholds}, 0, None
    start, end = overlap
    ref_name = max(cameras, key=lambda name: len(cameras[name]))
    ref_ts = cameras[ref_name]
    mask = (ref_ts >= start) & (ref_ts <= end)
    targets = ref_ts[mask]
    if len(targets) == 0:
        return {str(t): 0 for t in thresholds}, 0, end - start

    max_delta = np.zeros(len(targets), dtype=np.int64)
    for name, ts in cameras.items():
        if name == ref_name:
            continue
        max_delta = np.maximum(max_delta, nearest_abs_deltas(targets, ts))
    counts = {str(t): int(np.count_nonzero(max_delta <= t)) for t in thresholds}
    return counts, int(len(targets)), end - start


def count_pairwise_positives(
    cameras: dict[str, np.ndarray],
    thresholds: list[float],
) -> tuple[dict[str, int], int]:
    counts = {str(t): 0 for t in thresholds}
    total_candidates = 0
    for cam_a, cam_b in combinations(sorted(cameras), 2):
        ts_a = cameras[cam_a]
        ts_b = cameras[cam_b]
        start = max(int(ts_a[0]), int(ts_b[0]))
        end = min(int(ts_a[-1]), int(ts_b[-1]))
        if end <= start:
            continue
        source = ts_a[(ts_a >= start) & (ts_a <= end)]
        if len(source) == 0:
            continue
        deltas = nearest_abs_deltas(source, ts_b)
        total_candidates += int(len(source))
        for t in thresholds:
            counts[str(t)] += int(np.count_nonzero(deltas <= t))
    return counts, total_candidates


def side_of_episode(path: Path) -> str:
    return "human" if path.name.endswith("_human") else "robot"


def iter_episode_dirs(rgb_root: Path):
    for cfg_dir in sorted(rgb_root.glob("RH20T_cfg*")):
        if not cfg_dir.is_dir():
            continue
        for episode_dir in sorted(cfg_dir.glob("task_*")):
            if episode_dir.is_dir():
                yield cfg_dir.name, episode_dir


def process_episode(item: tuple[str, str, list[float]]) -> dict:
    cfg, episode_str, thresholds = item
    episode_dir = Path(episode_str)
    cameras = load_timestamps(episode_dir)
    side = side_of_episode(episode_dir)
    all_counts, all_candidates, overlap_ms = count_all_view_groups(cameras, thresholds)
    pair_counts, pair_candidates = count_pairwise_positives(cameras, thresholds)

    row = {
        "cfg": cfg,
        "episode": episode_dir.name,
        "side": side,
        "num_cameras": len(cameras),
        "overlap_ms": overlap_ms if overlap_ms is not None else "",
        "all_view_candidates": all_candidates,
        "pairwise_candidates": pair_candidates,
    }
    for t in thresholds:
        key = str(t)
        row[f"all_view_le_{key}ms"] = all_counts[key]
        row[f"all_view_ratio_le_{key}ms"] = (
            all_counts[key] / all_candidates if all_candidates else 0.0
        )
        row[f"pairwise_le_{key}ms"] = pair_counts[key]
        row[f"pairwise_ratio_le_{key}ms"] = (
            pair_counts[key] / pair_candidates if pair_candidates else 0.0
        )
    return row


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(args.thresholds_ms)

    rows = []
    totals: dict[str, dict[str, float]] = {}

    episode_items = [(cfg, str(episode_dir), thresholds) for cfg, episode_dir in iter_episode_dirs(args.rgb_root)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_episode, item) for item in episode_items]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            side = row["side"]
            all_candidates = int(row["all_view_candidates"])
            pair_candidates = int(row["pairwise_candidates"])
            total = totals.setdefault(
                side,
                {
                    "episodes": 0,
                    "cameras": 0,
                    "all_view_candidates": 0,
                    "pairwise_candidates": 0,
                    **{f"all_view_le_{t}ms": 0 for t in thresholds},
                    **{f"pairwise_le_{t}ms": 0 for t in thresholds},
                },
            )
            total["episodes"] += 1
            total["cameras"] += int(row["num_cameras"])
            total["all_view_candidates"] += all_candidates
            total["pairwise_candidates"] += pair_candidates
            for t in thresholds:
                total[f"all_view_le_{t}ms"] += int(row[f"all_view_le_{t}ms"])
                total[f"pairwise_le_{t}ms"] += int(row[f"pairwise_le_{t}ms"])

            if args.progress_every and idx % args.progress_every == 0:
                print(f"processed {idx} / {len(episode_items)} episodes", flush=True)

    rows.sort(key=lambda r: (r["cfg"], r["episode"]))

    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "episode_alignment_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"thresholds_ms": thresholds, "num_episodes": len(rows), "by_side": totals}
    for side, total in totals.items():
        for t in thresholds:
            all_den = total["all_view_candidates"]
            pair_den = total["pairwise_candidates"]
            total[f"all_view_ratio_le_{t}ms"] = (
                total[f"all_view_le_{t}ms"] / all_den if all_den else 0.0
            )
            total[f"pairwise_ratio_le_{t}ms"] = (
                total[f"pairwise_le_{t}ms"] / pair_den if pair_den else 0.0
            )
        total["mean_cameras_per_episode"] = (
            total["cameras"] / total["episodes"] if total["episodes"] else 0.0
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
