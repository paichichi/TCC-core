#!/usr/bin/env python3
"""Partition a follow-up grid across prd03 H200 workers and the RTX 5090."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PARAM_FIELDS = [
    "run_name",
    "backbone",
    "rho",
    "epsilon",
    "lr",
    "max_forward_step",
    "lambda_q",
    "lambda_sa",
    "lambda_mv",
]
ASSIGNMENT_FIELDS = [
    *PARAM_FIELDS,
    "upstream_site",
    "upstream_reused",
    "lite_site",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path}: no candidates")
    missing = set(PARAM_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len({row["run_name"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate run_name")
    for row in rows:
        if row["backbone"] not in {"vit_imagenet", "r3m_bn_bi"}:
            raise ValueError(f"{path}: unsupported backbone {row['backbone']!r}")
    if {row["backbone"] for row in rows} != {"vit_imagenet", "r3m_bn_bi"}:
        raise ValueError(f"{path}: both backbone families are required")
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--local-family",
        required=True,
        choices=("vit_imagenet", "r3m_bn_bi"),
    )
    parser.add_argument("--h200-output", required=True, type=Path)
    parser.add_argument("--local-output", required=True, type=Path)
    parser.add_argument("--assignment-output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_manifest(args.input)
    local_index = next(
        index
        for index, row in enumerate(rows)
        if row["backbone"] == args.local_family
    )
    local_rows = [rows[local_index]]
    h200_rows = [row for index, row in enumerate(rows) if index != local_index]

    assignments = []
    for index, row in enumerate(rows):
        if index == local_index:
            site = "rtx5090"
        elif row["backbone"] == "vit_imagenet":
            site = "prd03_h200_vit"
        else:
            site = "prd03_h200_r3m"
        assignments.append(
            {
                **row,
                "upstream_site": site,
                "upstream_reused": "false",
                "lite_site": "rtx5090",
            }
        )

    write_tsv(args.h200_output, h200_rows, PARAM_FIELDS)
    write_tsv(args.local_output, local_rows, PARAM_FIELDS)
    write_tsv(args.assignment_output, assignments, ASSIGNMENT_FIELDS)


if __name__ == "__main__":
    main()
