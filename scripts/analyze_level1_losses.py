#!/usr/bin/env python3
"""Level-1 loss-curve analysis for RH20T multi-view training runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = [
    "loss_total",
    "loss_softdtw",
    "loss_aux",
    "softdtw_top1",
    "softdtw_pos_dist",
    "softdtw_off_dist",
    "aux_top1_h",
    "aux_top1_r",
    "emb_std",
    "seconds",
    "cuda_mem_mb",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "run_dirs",
      nargs="+",
      type=str,
      help="Run directories containing losses.csv and optionally losses_rank1.csv.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=Path("downstream/analysis/level1_losses"),
      help="Directory for summary CSVs and plots.",
  )
  parser.add_argument(
      "--tail-steps",
      type=int,
      default=1000,
      help="Number of last steps used for stable tail statistics.",
  )
  parser.add_argument(
      "--smooth",
      type=int,
      default=200,
      help="Moving-average window for plots. Use 1 to disable smoothing.",
  )
  parser.add_argument(
      "--no-plots",
      action="store_true",
      help="Only write CSV summaries.",
  )
  return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
  """Resolve Linux paths and common Windows drive paths used from WSL."""
  text = raw_path.strip().strip("\"'")
  if len(text) >= 3 and text[1:3] in {":\\", ":/"}:
    drive = text[0].lower()
    rest = text[3:].replace("\\", "/")
    return Path(f"/mnt/{drive}") / rest
  return Path(text)


def read_loss_csv(path: Path) -> list[dict[str, float | int]]:
  rows = []
  with path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      parsed: dict[str, float | int] = {}
      for key, value in row.items():
        if key == "step":
          parsed[key] = int(value)
        else:
          try:
            parsed[key] = float(value)
          except (TypeError, ValueError):
            parsed[key] = math.nan
      rows.append(parsed)
  return rows


def mean(values: list[float]) -> float:
  values = [value for value in values if math.isfinite(value)]
  return sum(values) / len(values) if values else math.nan


def tail_rows(rows: list[dict[str, float | int]], tail_steps: int):
  if not rows:
    return []
  max_step = int(rows[-1]["step"])
  min_step = max(1, max_step - tail_steps + 1)
  return [row for row in rows if int(row["step"]) >= min_step]


def summarize_file(
    run_name: str,
    rank_name: str,
    rows: list[dict[str, float | int]],
    tail_steps: int,
) -> dict[str, str | int | float]:
  tail = tail_rows(rows, tail_steps)
  final = rows[-1]
  out: dict[str, str | int | float] = {
      "run": run_name,
      "rank": rank_name,
      "steps": len(rows),
      "final_step": int(final["step"]),
  }
  for metric in DEFAULT_METRICS:
    if metric in final:
      out[f"final_{metric}"] = final[metric]
      out[f"tail_mean_{metric}"] = mean([float(row[metric]) for row in tail])
  if "softdtw_pos_dist" in final and "softdtw_off_dist" in final:
    out["final_softdtw_margin"] = (
        float(final["softdtw_off_dist"]) - float(final["softdtw_pos_dist"])
    )
    out["tail_mean_softdtw_margin"] = mean([
        float(row["softdtw_off_dist"]) - float(row["softdtw_pos_dist"])
        for row in tail
    ])
  return out


def write_table(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  fieldnames = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def moving_average(values: list[float], window: int) -> list[float]:
  if window <= 1:
    return values
  out = []
  running = 0.0
  queue = []
  for value in values:
    queue.append(value)
    running += value
    if len(queue) > window:
      running -= queue.pop(0)
    out.append(running / len(queue))
  return out


def plot_metrics(
    all_rows: dict[tuple[str, str], list[dict[str, float | int]]],
    out_dir: Path,
    smooth: int,
) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib is not installed; skipping plots.")
    return

  plot_dir = out_dir / "plots"
  plot_dir.mkdir(parents=True, exist_ok=True)

  for metric in DEFAULT_METRICS:
    plt.figure(figsize=(9, 5))
    plotted = False
    for (run_name, rank_name), rows in sorted(all_rows.items()):
      if not rows or metric not in rows[0]:
        continue
      steps = [int(row["step"]) for row in rows]
      values = [float(row[metric]) for row in rows]
      values = moving_average(values, smooth)
      plt.plot(steps, values, label=f"{run_name}/{rank_name}", linewidth=1.4)
      plotted = True
    if plotted:
      plt.xlabel("step")
      plt.ylabel(metric)
      plt.title(metric)
      plt.grid(True, alpha=0.25)
      plt.legend(fontsize=8)
      plt.tight_layout()
      plt.savefig(plot_dir / f"{metric}.png", dpi=160)
    plt.close()

  for run_name in sorted({key[0] for key in all_rows}):
    plt.figure(figsize=(9, 5))
    plotted = False
    for rank_name in ["rank0", "rank1"]:
      rows = all_rows.get((run_name, rank_name), [])
      if not rows:
        continue
      steps = [int(row["step"]) for row in rows]
      pos = moving_average(
          [float(row["softdtw_pos_dist"]) for row in rows], smooth)
      off = moving_average(
          [float(row["softdtw_off_dist"]) for row in rows], smooth)
      plt.plot(steps, pos, label=f"{rank_name} pos", linewidth=1.4)
      plt.plot(steps, off, label=f"{rank_name} off", linewidth=1.4)
      plotted = True
    if plotted:
      plt.xlabel("step")
      plt.ylabel("Soft-DTW distance")
      plt.title(f"{run_name}: positive vs negative distance")
      plt.grid(True, alpha=0.25)
      plt.legend(fontsize=8)
      plt.tight_layout()
      plt.savefig(plot_dir / f"{run_name}_softdtw_pos_off.png", dpi=160)
    plt.close()


def main() -> None:
  args = parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  all_rows: dict[tuple[str, str], list[dict[str, float | int]]] = {}
  summary_rows = []

  for raw_run_dir in args.run_dirs:
    run_dir = resolve_path(raw_run_dir)
    run_name = run_dir.name
    rank_files = [
        ("rank0", run_dir / "losses.csv"),
        ("rank1", run_dir / "losses_rank1.csv"),
    ]
    for rank_name, loss_path in rank_files:
      if not loss_path.exists():
        print(f"missing: {loss_path}")
        continue
      rows = read_loss_csv(loss_path)
      all_rows[(run_name, rank_name)] = rows
      summary_rows.append(
          summarize_file(run_name, rank_name, rows, args.tail_steps))

  write_table(args.out_dir / "rank_summary.csv", summary_rows)

  combined = defaultdict(list)
  for row in summary_rows:
    combined[row["run"]].append(row)
  combined_rows = []
  for run_name, rows in sorted(combined.items()):
    out = {"run": run_name, "ranks": len(rows)}
    numeric_keys = [
        key for key in rows[0]
        if key not in {"run", "rank"} and isinstance(rows[0][key], (int, float))
    ]
    for key in numeric_keys:
      out[f"mean_{key}"] = mean([float(row[key]) for row in rows])
    combined_rows.append(out)
  write_table(args.out_dir / "run_summary.csv", combined_rows)

  if not args.no_plots:
    plot_metrics(all_rows, args.out_dir, args.smooth)

  print(f"wrote: {args.out_dir / 'rank_summary.csv'}")
  print(f"wrote: {args.out_dir / 'run_summary.csv'}")
  if not args.no_plots:
    print(f"wrote plots under: {args.out_dir / 'plots'}")


if __name__ == "__main__":
  main()
