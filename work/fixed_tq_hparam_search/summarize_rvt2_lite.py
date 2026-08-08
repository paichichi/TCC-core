#!/usr/bin/env python3
"""Validate one 3x12 RVT2-lite evaluation and update the central result table."""

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev


LEVEL1 = {
    "close_fridge",
    "close_microwave",
    "close_laptop_lid",
    "toilet_seat_down",
    "open_grill",
    "phone_on_base",
}
LEVEL2 = {
    "take_usb_out_of_computer",
    "take_lid_off_saucepan",
    "turn_oven_on",
    "beat_the_buzz",
    "water_plants",
    "unplug_charger",
}
FIELDS = [
    "run_name",
    "backbone",
    "repeats",
    "tasks_per_repeat",
    "episodes_per_task",
    "overall_mean",
    "overall_std",
    "level1_mean",
    "level1_std",
    "level2_mean",
    "level2_std",
    "raw_csv",
]


def mean_std(values: list[float]) -> tuple[float, float]:
    return fmean(values), pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--raw-run-name",
        help="Run label stored in the raw CSV when importing a legacy result.",
    )
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--central", required=True, type=Path)
    parser.add_argument("--task-summary", required=True, type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    args = parser.parse_args()
    raw_run_name = args.raw_run_name or args.run_name

    with args.raw.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("status") == "ok"]
    if len(rows) != 36:
        raise ValueError(f"Expected 36 successful task-run rows, found {len(rows)}")

    by_repeat: dict[int, dict[str, float]] = defaultdict(dict)
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["run"] != raw_run_name:
            raise ValueError(f"Unexpected run name in raw results: {row['run']}")
        repeat = int(row["repeat"])
        task = row["task"]
        if task not in LEVEL1 | LEVEL2:
            raise ValueError(f"Unexpected task: {task}")
        if task in by_repeat[repeat]:
            raise ValueError(f"Duplicate task in repeat {repeat}: {task}")
        score = float(row["success_rate"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite success rate: {row}")
        by_repeat[repeat][task] = score
        by_task[task].append(score)

    if set(by_repeat) != {1, 2, 3}:
        raise ValueError(f"Expected repeats 1,2,3; found {sorted(by_repeat)}")
    expected_tasks = LEVEL1 | LEVEL2
    for repeat, task_scores in by_repeat.items():
        if set(task_scores) != expected_tasks:
            missing = sorted(expected_tasks - set(task_scores))
            extra = sorted(set(task_scores) - expected_tasks)
            raise ValueError(
                f"Repeat {repeat} task mismatch: missing={missing}, extra={extra}"
            )

    overall_by_repeat = [
        fmean(task_scores.values()) for task_scores in by_repeat.values()
    ]
    level1_by_repeat = [
        fmean(task_scores[task] for task in LEVEL1)
        for task_scores in by_repeat.values()
    ]
    level2_by_repeat = [
        fmean(task_scores[task] for task in LEVEL2)
        for task_scores in by_repeat.values()
    ]
    overall_mean, overall_std = mean_std(overall_by_repeat)
    level1_mean, level1_std = mean_std(level1_by_repeat)
    level2_mean, level2_std = mean_std(level2_by_repeat)

    args.task_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.task_summary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "level", "runs", "mean_success_rate", "std_success_rate"])
        for task in sorted(by_task):
            mean, std = mean_std(by_task[task])
            writer.writerow(
                [task, "level1" if task in LEVEL1 else "level2", 3, f"{mean:.6f}", f"{std:.6f}"]
            )

    result = {
        "run_name": args.run_name,
        "backbone": args.backbone,
        "repeats": 3,
        "tasks_per_repeat": 12,
        "episodes_per_task": args.episodes_per_task,
        "overall_mean": f"{overall_mean:.6f}",
        "overall_std": f"{overall_std:.6f}",
        "level1_mean": f"{level1_mean:.6f}",
        "level1_std": f"{level1_std:.6f}",
        "level2_mean": f"{level2_mean:.6f}",
        "level2_std": f"{level2_std:.6f}",
        "raw_csv": str(args.raw.resolve()),
    }

    args.central.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.central.with_suffix(args.central.suffix + ".lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        existing: list[dict[str, str]] = []
        if args.central.exists():
            with args.central.open(newline="") as handle:
                existing = list(csv.DictReader(handle))
        existing = [row for row in existing if row["run_name"] != args.run_name]
        existing.append(result)
        existing.sort(key=lambda row: row["run_name"])
        temporary = args.central.with_suffix(args.central.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing)
        os.replace(temporary, args.central)


if __name__ == "__main__":
    main()
