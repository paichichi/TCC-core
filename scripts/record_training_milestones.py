#!/usr/bin/env python3
"""Record fixed training milestones for the resumed single/multi runs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path


METRICS = (
    "loss_total",
    "loss_softdtw",
    "loss_align",
    "loss_struct",
    "softdtw_top1",
    "control_gain_gap",
    "control_win_rate",
    "adapter_grad_norm",
    "head_grad_norm",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--run",
      action="append",
      required=True,
      metavar="LABEL=RUN_DIR",
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--milestones", type=int, nargs="+", default=[25000, 30000, 35000, 40000])
  parser.add_argument("--window", type=int, default=500)
  parser.add_argument("--poll-seconds", type=int, default=60)
  return parser.parse_args()


def load_existing(path: Path) -> dict[tuple[str, int], dict[str, object]]:
  if not path.is_file():
    return {}
  with path.open(newline="", encoding="utf-8") as handle:
    return {
        (row["run"], int(row["milestone"])): row
        for row in csv.DictReader(handle)
    }


def summarize(label: str, run_dir: Path, milestone: int, window: int) -> dict[str, object] | None:
  checkpoint = run_dir / f"checkpoint_{milestone:06d}.pt"
  csv_path = run_dir / "losses.csv"
  if not checkpoint.is_file() or not csv_path.is_file():
    return None
  with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  selected = [row for row in rows if int(row["step"]) <= milestone][-window:]
  if not selected or int(selected[-1]["step"]) != milestone:
    return None
  result: dict[str, object] = {
      "run": label,
      "milestone": milestone,
      "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
      "window_start": int(selected[0]["step"]),
      "window_end": int(selected[-1]["step"]),
      "window_rows": len(selected),
      "checkpoint": str(checkpoint),
  }
  for metric in METRICS:
    result[f"{metric}_mean"] = sum(float(row[metric]) for row in selected) / len(selected)
    result[f"{metric}_at_step"] = float(selected[-1][metric])
  return result


def write_outputs(path: Path, records: dict[tuple[str, int], dict[str, object]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  ordered = [records[key] for key in sorted(records)]
  fields = [
      "run", "milestone", "recorded_at", "window_start", "window_end",
      "window_rows", "checkpoint",
  ]
  for metric in METRICS:
    fields.extend((f"{metric}_mean", f"{metric}_at_step"))
  temp_csv = path.with_suffix(path.suffix + ".tmp")
  with temp_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(ordered)
  temp_csv.replace(path)
  json_path = path.with_suffix(".json")
  temp_json = json_path.with_suffix(json_path.suffix + ".tmp")
  temp_json.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
  temp_json.replace(json_path)


def main() -> None:
  args = parse_args()
  runs: dict[str, Path] = {}
  for spec in args.run:
    label, raw_path = spec.split("=", 1)
    runs[label] = Path(raw_path)
  targets = {(label, milestone) for label in runs for milestone in args.milestones}
  records = load_existing(args.output)
  while not targets.issubset(records):
    changed = False
    for label, milestone in sorted(targets - records.keys()):
      record = summarize(label, runs[label], milestone, args.window)
      if record is not None:
        records[(label, milestone)] = record
        changed = True
        print(
            f"[{record['recorded_at']}] recorded {label} step={milestone}",
            flush=True,
        )
    if changed:
      write_outputs(args.output, records)
    if not targets.issubset(records):
      time.sleep(args.poll_seconds)


if __name__ == "__main__":
  main()
