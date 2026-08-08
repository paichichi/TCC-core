#!/usr/bin/env python3
"""Summarize comparable 40k upstream runs without ranking by raw loss alone."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from statistics import fmean, pstdev


METRICS = (
    "loss_total",
    "loss_softdtw",
    "loss_align",
    "loss_struct",
    "loss_aux",
    "soft_alignment_teacher_entropy",
    "soft_alignment_kl",
    "soft_alignment_pred_entropy",
    "soft_alignment_teacher_expected_deviation",
    "soft_alignment_teacher_diag_mass",
    "softdtw_top1",
    "emb_std",
    "adapter_grad_norm",
    "head_grad_norm",
)


def finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def metric_summary(
    rows: list[dict[str, str]], metric: str, max_step: int
) -> dict[str, float | None]:
    points = [
        (float(row["step"]), value)
        for row in rows
        if (value := finite_float(row.get(metric))) is not None
    ]
    if not points:
        return {}

    early_limit = min(1000, max_step)
    late_start = max(1, max_step - 999)
    last_10k_start = max(1, max_step - 9999)
    early = [value for step, value in points if step <= early_limit]
    late = [value for step, value in points if step >= late_start]
    last_10k = [(step, value) for step, value in points if step >= last_10k_start]

    first_mean = fmean(early) if early else None
    last_mean = fmean(late) if late else None
    all_slope = linear_slope(
        [step for step, _ in points], [value for _, value in points]
    )
    late_slope = linear_slope(
        [step for step, _ in last_10k], [value for _, value in last_10k]
    )
    return {
        "first1k_mean": first_mean,
        "last1k_mean": last_mean,
        "delta_last_minus_first": (
            last_mean - first_mean
            if first_mean is not None and last_mean is not None
            else None
        ),
        "slope_all_per10k": all_slope * 10000 if all_slope is not None else None,
        "slope_last10k_per10k": (
            late_slope * 10000 if late_slope is not None else None
        ),
        "last1k_std": pstdev(late) if len(late) > 1 else 0.0,
    }


def load_optional_lite_results(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["run_name"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-step", type=int, default=40000)
    parser.add_argument("--lite-results", type=Path)
    args = parser.parse_args()

    lite_results = load_optional_lite_results(args.lite_results)
    with args.manifest.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

    summaries: list[dict[str, object]] = []
    for specification in manifest_rows:
        run_name = specification["run_name"]
        if not run_name:
            continue
        run_dir = args.output_root / run_name
        losses_path = run_dir / "losses.csv"
        checkpoint = run_dir / f"checkpoint_{args.expected_step:06d}.pt"
        summary: dict[str, object] = {
            **specification,
            "run_dir": str(run_dir),
            "status": "missing",
            "max_step": 0,
            "num_logged_steps": 0,
            "checkpoint_complete": checkpoint.exists(),
        }

        if losses_path.exists():
            with losses_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows = [row for row in rows if row.get("step", "").isdigit()]
            max_step = max((int(row["step"]) for row in rows), default=0)
            summary["max_step"] = max_step
            summary["num_logged_steps"] = len(rows)
            summary["status"] = (
                "complete"
                if max_step >= args.expected_step and checkpoint.exists()
                else "partial"
            )
            for metric in METRICS:
                for suffix, value in metric_summary(rows, metric, max_step).items():
                    summary[f"{metric}__{suffix}"] = value

        if run_name in lite_results:
            for key, value in lite_results[run_name].items():
                if key != "run_name":
                    summary[f"rvt2_lite__{key}"] = value
        summaries.append(summary)

    fieldnames: list[str] = []
    for row in summaries:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, args.output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
