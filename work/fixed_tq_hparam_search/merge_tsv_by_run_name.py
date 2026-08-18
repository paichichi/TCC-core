#!/usr/bin/env python3
"""Atomically merge TSV manifests by run_name with conflict detection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        rows = list(reader)
    if "run_name" not in fields:
        raise ValueError(f"{path}: run_name column is required")
    if len({row["run_name"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate run_name")
    return fields, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--additions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fields, base_rows = read(args.base)
    addition_fields, additions = read(args.additions)
    if fields != addition_fields:
        raise ValueError(
            f"Schema mismatch: base={fields!r}, additions={addition_fields!r}"
        )

    merged = {row["run_name"]: row for row in base_rows}
    order = [row["run_name"] for row in base_rows]
    for row in additions:
        run_name = row["run_name"]
        if run_name in merged:
            if row != merged[run_name]:
                raise ValueError(f"Conflicting duplicate run_name: {run_name}")
            continue
        merged[run_name] = row
        order.append(run_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(merged[run_name] for run_name in order)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
