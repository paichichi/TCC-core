#!/usr/bin/env python3
"""Build the final staged hyperparameter-search CSV and Markdown report."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


STAGES = ("A1", "A2", "B", "C")
BACKBONES = ("vit_imagenet", "r3m_bn_bi")
WINNER_FIELDS = [
    "stage_label",
    "backbone",
    "selected_run_name",
    "rho",
    "epsilon",
    "lr",
    "max_forward_step",
    "lambda_q",
    "lambda_sa",
    "lambda_mv",
    "overall_mean",
    "overall_std",
    "level1_mean",
    "level1_std",
    "level2_mean",
    "level2_std",
    "runner_up_gap",
    "pooled_standard_error",
    "decisive_at_one_se",
]


def read(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def metric(row: dict[str, str], prefix: str) -> str:
    return f"{float(row[prefix + '_mean']):.2f} ± {float(row[prefix + '_std']):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--lite-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    winners: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    for stage in STAGES:
        selection_path = args.state_dir / f"{stage}_selection.tsv"
        audit_path = args.state_dir / f"{stage}_selection_audit.tsv"
        selection = read(selection_path)
        audit = read(audit_path)
        if {row["backbone"] for row in selection} != set(BACKBONES):
            raise ValueError(f"{selection_path}: expected both backbones")
        if len(selection) != 2:
            raise ValueError(f"{selection_path}: expected exactly two winners")
        winners.extend(selection)
        audit_rows.extend(audit)

    lite_rows = read(args.lite_results, delimiter=",")
    lite_by_name = {row["run_name"]: row for row in lite_rows}
    missing = [
        row["selected_run_name"]
        for row in winners
        if row["selected_run_name"] not in lite_by_name
    ]
    if missing:
        raise ValueError(f"Winner lite results are missing: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.provenance is not None:
        if not args.provenance.is_file():
            raise ValueError(f"Provenance file is missing: {args.provenance}")
        shutil.copy2(
            args.provenance,
            args.output_dir / "experiment_provenance.yaml",
        )
    write_csv(args.output_dir / "stage_winners.csv", winners, WINNER_FIELDS)
    write_csv(
        args.output_dir / "stage_candidate_audit.csv",
        audit_rows,
        list(audit_rows[0]),
    )
    write_csv(
        args.output_dir / "all_rvt2_lite_results.csv",
        lite_rows,
        list(lite_rows[0]),
    )
    final_winners = [
        row for row in winners if row["stage_label"] == "C"
    ]
    write_csv(
        args.output_dir / "final_winners.csv",
        final_winners,
        WINNER_FIELDS,
    )

    lines = [
        "# Dual-backbone hyperparameter search",
        "",
        "All selections use the fixed RVT2-lite protocol "
        "(3 runs × 12 tasks × 10 episodes).",
        "",
        "Exact source, dataset-index, checkpoint, and software hashes are "
        "recorded in `experiment_provenance.yaml`.",
        "",
        "## Stage winners",
        "",
        "| Stage | Backbone | ρ | ε | LR | Q step | λQ | λSA | λInfoNCE | Overall | Level-1 | Level-2 | >1 SE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in winners:
        lines.append(
            "| {stage_label} | {backbone} | {rho} | {epsilon} | {lr} | "
            "{max_forward_step} | {lambda_q} | {lambda_sa} | {lambda_mv} | "
            "{overall} | {level1} | {level2} | {decisive} |".format(
                **row,
                overall=metric(row, "overall"),
                level1=metric(row, "level1"),
                level2=metric(row, "level2"),
                decisive="yes"
                if row["decisive_at_one_se"].lower() == "true"
                else "no",
            )
        )

    lines.extend(["", "## Final selected configurations", ""])
    for row in final_winners:
        lines.extend(
            [
                f"### {row['backbone']}",
                "",
                f"- Run: `{row['selected_run_name']}`",
                f"- ρ={row['rho']}, ε={row['epsilon']}, lr={row['lr']}",
                (
                    f"- max_forward_step={row['max_forward_step']}, "
                    f"λQ={row['lambda_q']}, λSA={row['lambda_sa']}, "
                    f"λInfoNCE={row['lambda_mv']}"
                ),
                f"- Overall: {metric(row, 'overall')}",
                f"- Level-1: {metric(row, 'level1')}",
                f"- Level-2: {metric(row, 'level2')}",
                "",
            ]
        )

    calibration_path = (
        args.state_dir / "stageA_platform_calibration_summary.csv"
    )
    if calibration_path.exists():
        calibration = read(calibration_path, delimiter=",")
        lines.extend(
            [
                "## H200 − RTX 5090 platform calibration",
                "",
                "| Backbone | Overall Δ | Level-1 Δ | Level-2 Δ |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in calibration:
            lines.append(
                "| {backbone} | {overall:.2f} | {level1:.2f} | {level2:.2f} |".format(
                    backbone=row["backbone"],
                    overall=float(row["h200_minus_rtx5090_overall"]),
                    level1=float(row["h200_minus_rtx5090_level1"]),
                    level2=float(row["h200_minus_rtx5090_level2"]),
                )
            )
        lines.append("")

    report = args.output_dir / "final_report.md"
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(report)


if __name__ == "__main__":
    main()
