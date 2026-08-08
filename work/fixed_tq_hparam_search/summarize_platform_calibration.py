#!/usr/bin/env python3
"""Summarize matched H200 versus RTX 5090 RVT2-lite calibration anchors."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


OUTPUT_FIELDS = [
    "backbone",
    "rho",
    "epsilon",
    "lr",
    "rtx5090_run",
    "prd03_h200_run",
    "rtx5090_overall_mean",
    "h200_overall_mean",
    "h200_minus_rtx5090_overall",
    "pooled_overall_std",
    "rtx5090_level1_mean",
    "h200_level1_mean",
    "h200_minus_rtx5090_level1",
    "pooled_level1_std",
    "rtx5090_level2_mean",
    "h200_level2_mean",
    "h200_minus_rtx5090_level2",
    "pooled_level2_std",
]


def read(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def number(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}: non-finite {field}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--lite-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    calibration = read(args.calibration, "\t")
    results = read(args.lite_results, ",")
    by_name = {row["run_name"]: row for row in results}
    output = []

    for pair in calibration:
        local_name = pair["rtx5090_run"]
        h200_name = pair["prd03_h200_run"]
        missing = [name for name in (local_name, h200_name) if name not in by_name]
        if missing:
            raise ValueError(f"Calibration is incomplete; missing: {missing}")
        local = by_name[local_name]
        h200 = by_name[h200_name]
        for result, name in ((local, local_name), (h200, h200_name)):
            if (
                result.get("repeats"),
                result.get("tasks_per_repeat"),
                result.get("episodes_per_task"),
            ) != ("3", "12", "10"):
                raise ValueError(f"{name}: incompatible RVT2-lite protocol")

        row = {field: pair[field] for field in ("backbone", "rho", "epsilon", "lr")}
        row.update(
            {
                "rtx5090_run": local_name,
                "prd03_h200_run": h200_name,
            }
        )
        for short, prefix in (
            ("overall", "overall"),
            ("level1", "level1"),
            ("level2", "level2"),
        ):
            local_mean = number(local, f"{prefix}_mean", local_name)
            h200_mean = number(h200, f"{prefix}_mean", h200_name)
            local_std = number(local, f"{prefix}_std", local_name)
            h200_std = number(h200, f"{prefix}_std", h200_name)
            row[f"rtx5090_{short}_mean"] = local_mean
            row[f"h200_{short}_mean"] = h200_mean
            row[f"h200_minus_rtx5090_{short}"] = h200_mean - local_mean
            row[f"pooled_{short}_std"] = math.sqrt(local_std**2 + h200_std**2)
        output.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
