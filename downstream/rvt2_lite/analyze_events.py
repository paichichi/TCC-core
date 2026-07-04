#!/usr/bin/env python3
"""Summarize RVT2-lite TensorBoard event scalars."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_RUNS = ["d4r", "hrp", "ours_6ts4v", "ours_8ts3v"]
METRICS = [
    "total_loss",
    "trans_loss",
    "rot_loss_x",
    "rot_loss_y",
    "rot_loss_z",
    "grip_loss",
    "collision_loss",
    "lr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("downstream/rvt2_lite/runs/base"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("downstream/rvt2_lite/runs"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    return parser.parse_args()


def read_scalars(run_dir: Path) -> dict[str, float]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found under {run_dir}")

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    out = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        key = tag.removeprefix("train_")
        out[key] = float(events[-1].value)
    return out


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bars(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    runs = [str(row["run"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ["total_loss", "trans_loss"]):
        values = [float(row[metric]) for row in rows]
        axis.bar(runs, values, color=["#6b7280", "#2563eb", "#0f766e", "#9333ea"])
        axis.set_title(metric)
        axis.set_ylabel("loss")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = []
    for run in args.runs:
        scalars = read_scalars(args.runs_root / run)
        row = {"run": run}
        for metric in METRICS:
            row[metric] = scalars.get(metric, "")
        rows.append(row)

    summary_path = args.out_dir / "event_summary.csv"
    plot_path = args.out_dir / "event_loss_bars.png"
    write_summary(summary_path, rows)
    plot_bars(plot_path, rows)
    print(f"wrote: {summary_path}")
    print(f"wrote: {plot_path}")


if __name__ == "__main__":
    main()
