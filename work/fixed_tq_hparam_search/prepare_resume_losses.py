#!/usr/bin/env python3
"""Make losses.csv consistent with the latest saved checkpoint before resume."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint_step = int(args.checkpoint.stem.rsplit("_", 1)[1])
    losses_path = args.run_dir / "losses.csv"
    if not losses_path.exists():
        return

    with losses_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise RuntimeError(f"{losses_path} has no step column")
        rows = list(reader)
        fieldnames = reader.fieldnames

    kept = [row for row in rows if int(row["step"]) <= checkpoint_step]
    if len(kept) == len(rows):
        return
    if not kept or int(kept[-1]["step"]) != checkpoint_step:
        raise RuntimeError(
            f"{losses_path} has no row matching checkpoint step {checkpoint_step}"
        )

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = losses_path.with_name(f"losses.csv.pre-resume-{stamp}")
    shutil.copy2(losses_path, backup)

    temporary = losses_path.with_suffix(".csv.truncate.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(temporary, losses_path)


if __name__ == "__main__":
    main()

