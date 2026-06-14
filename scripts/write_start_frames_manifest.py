#!/usr/bin/env python3
"""Write tuned RH20T human start frames into manifest.csv.

This updates metadata only. It does not move, delete, trim, or re-encode image
frames. Human rows get a tuned visual onset; robot rows get start_frame=0.
"""

import argparse
import csv
import os
import shutil
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import cv2


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default="/home/paichichi/data/RH20T/TCC_RH20T")
  parser.add_argument("--workers", type=int, default=12)
  parser.add_argument("--max-scan-frames", type=int, default=240)
  parser.add_argument("--audit-name", default="start_frame_audit.csv")
  parser.add_argument("--limit", type=int, default=0)
  return parser.parse_args()


def numeric_jpgs(seq_dir):
  return sorted(seq_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def small_gray(path):
  bgr = cv2.imread(str(path))
  if bgr is None:
    raise RuntimeError(f"Failed to read frame: {path}")
  gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
  return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def tile_max(diff, tile=10):
  h, w = diff.shape
  max_value = 0.0
  for y in range(0, h, tile):
    for x in range(0, w, tile):
      value = float(diff[y:min(h, y + tile), x:min(w, x + tile)].mean())
      max_value = max(max_value, value)
  return max_value


def stable_first(values, threshold, window=4, min_hits=3):
  if not values:
    return ""
  for start in range(0, max(1, len(values) - window + 1)):
    chunk = values[start:start + window]
    if len(chunk) < window:
      break
    if sum(value >= threshold for value in chunk) >= min_hits:
      return start
  return ""


def tuned_visual_onset(paths):
  grays = [small_gray(path) for path in paths]
  if not grays:
    return {
        "start_frame": 0,
        "source": "empty_default_0",
        "motion_onset_frame": "",
        "strict_motion_onset_frame": "",
        "local_baseline_onset_frame": "",
        "mean_baseline_onset_frame": "",
        "mean_baseline_threshold": "",
        "scanned_frames": 0,
    }

  base = grays[0]
  base_mean = []
  base_tile_max = []
  adj_mean = []
  adj_tile_max = []

  for idx, gray in enumerate(grays):
    base_diff = cv2.absdiff(base, gray)
    base_mean.append(float(base_diff.mean()))
    base_tile_max.append(tile_max(base_diff))
    if idx == 0:
      adj_mean.append(0.0)
      adj_tile_max.append(0.0)
    else:
      adj_diff = cv2.absdiff(grays[idx - 1], gray)
      adj_mean.append(float(adj_diff.mean()))
      adj_tile_max.append(tile_max(adj_diff))

  first10 = base_mean[:min(10, len(base_mean))]
  med = statistics.median(first10) if first10 else 0.0
  mad = statistics.median([abs(x - med) for x in first10]) if first10 else 0.0

  mean_baseline_threshold = max(3.0, med + 5.0 * mad + 1.0)
  local_baseline_threshold = 8.0
  local_motion_threshold = 12.0
  sensitive_local_motion_threshold = 10.0
  mean_motion_threshold = 0.8

  mean_baseline_onset = stable_first(
      base_mean, mean_baseline_threshold, window=4, min_hits=3)
  local_baseline_onset = stable_first(
      base_tile_max, local_baseline_threshold, window=4, min_hits=3)
  strict_motion_onset = stable_first(
      adj_tile_max, local_motion_threshold, window=4, min_hits=3)
  sensitive_motion_onset = stable_first(
      adj_tile_max, sensitive_local_motion_threshold, window=4, min_hits=2)
  mean_motion_onset = stable_first(
      adj_mean, mean_motion_threshold, window=4, min_hits=2)

  candidates = []
  sources = []

  def add_candidate(name, value):
    if value != "":
      candidates.append(int(value))
      sources.append((name, int(value)))

  add_candidate("motion", strict_motion_onset)

  if local_baseline_onset != "":
    local = int(local_baseline_onset)
    mean = int(mean_baseline_onset) if mean_baseline_onset != "" else None
    sensitive_motion = (
        int(sensitive_motion_onset) if sensitive_motion_onset != "" else None)
    strict_motion = int(strict_motion_onset) if strict_motion_onset != "" else None
    mean_agrees = mean is not None and abs(mean - local) <= 8
    motion_agrees = (
        sensitive_motion is not None and local >= 30 and
        abs(sensitive_motion - local) <= 3)
    if (
        (local >= 20 and mean_agrees and
         (strict_motion is None or local <= strict_motion)) or
        motion_agrees or
        (strict_motion is None and local >= 30)
    ):
      add_candidate("local_baseline", local)

  if not candidates:
    add_candidate("mean_baseline", mean_baseline_onset)

  if candidates:
    start_frame = min(candidates)
    source = "+".join(name for name, value in sources if value == start_frame)
  else:
    start_frame = 0
    source = "missing_default_0"

  return {
      "start_frame": start_frame,
      "source": source,
      "motion_onset_frame": sensitive_motion_onset,
      "strict_motion_onset_frame": strict_motion_onset,
      "mean_motion_onset_frame": mean_motion_onset,
      "local_baseline_onset_frame": local_baseline_onset,
      "mean_baseline_onset_frame": mean_baseline_onset,
      "mean_baseline_threshold": f"{mean_baseline_threshold:.3f}",
      "local_baseline_threshold": f"{local_baseline_threshold:.3f}",
      "local_motion_threshold": f"{local_motion_threshold:.3f}",
      "mean_motion_threshold": f"{mean_motion_threshold:.3f}",
      "scanned_frames": len(paths),
  }


def process_human(task):
  root, max_scan_frames, row = task
  root = Path(root)
  sequence_id = row["sequence_id"]
  seq_dir = root / "train" / row["episode_name"] / sequence_id
  try:
    all_paths = numeric_jpgs(seq_dir)
    total_frames = len(all_paths)
    scan_paths = all_paths[:max_scan_frames]
    onset = tuned_visual_onset(scan_paths)
    start_frame = min(int(onset["start_frame"]), max(0, total_frames - 1))
    return {
        "sequence_id": sequence_id,
        "paired_sequence_id": row["paired_sequence_id"],
        "episode_id": row["episode_id"],
        "episode_name": row["episode_name"],
        "task_id": row["task_id"],
        "camera_id": row["camera_id"],
        "camera_view": row.get("camera_view", ""),
        "manifest_num_frames": row["num_frames"],
        "actual_num_frames": total_frames,
        "start_frame": start_frame,
        "source": onset["source"],
        "scanned_frames": onset["scanned_frames"],
        "motion_onset_frame": onset.get("motion_onset_frame", ""),
        "strict_motion_onset_frame": onset.get("strict_motion_onset_frame", ""),
        "mean_motion_onset_frame": onset.get("mean_motion_onset_frame", ""),
        "local_baseline_onset_frame": onset.get("local_baseline_onset_frame", ""),
        "mean_baseline_onset_frame": onset.get("mean_baseline_onset_frame", ""),
        "mean_baseline_threshold": onset.get("mean_baseline_threshold", ""),
        "local_baseline_threshold": onset.get("local_baseline_threshold", ""),
        "local_motion_threshold": onset.get("local_motion_threshold", ""),
        "mean_motion_threshold": onset.get("mean_motion_threshold", ""),
        "status": "ok",
        "error": "",
      }
  except Exception as exc:
    return {
        "sequence_id": sequence_id,
        "paired_sequence_id": row["paired_sequence_id"],
        "episode_id": row["episode_id"],
        "episode_name": row["episode_name"],
        "task_id": row["task_id"],
        "camera_id": row["camera_id"],
        "camera_view": row.get("camera_view", ""),
        "manifest_num_frames": row["num_frames"],
        "actual_num_frames": "",
        "start_frame": 0,
        "source": "error_default_0",
        "scanned_frames": 0,
        "motion_onset_frame": "",
        "strict_motion_onset_frame": "",
        "mean_motion_onset_frame": "",
        "local_baseline_onset_frame": "",
        "mean_baseline_onset_frame": "",
        "mean_baseline_threshold": "",
        "local_baseline_threshold": "",
        "local_motion_threshold": "",
        "mean_motion_threshold": "",
        "status": "error",
        "error": str(exc),
    }


def read_lookup(root):
  by_human = {}
  with (root / "lookup.csv").open(newline="") as fp:
    for row in csv.DictReader(fp):
      by_human[row["human_sequence_id"]] = row
  return by_human


def build_human_tasks(root, manifest_rows, lookup, max_scan_frames, limit):
  tasks = []
  for row in manifest_rows:
    if row["role"] != "h":
      continue
    info = lookup.get(row["sequence_id"], {})
    episode_name = info.get("class_name", f"episode_{int(row['episode_id']):06d}")
    task = dict(row)
    task["episode_name"] = episode_name
    task["camera_view"] = info.get("camera_view", "")
    tasks.append((str(root), max_scan_frames, task))
    if limit and len(tasks) >= limit:
      break
  return tasks


def write_audit(path, audit_rows):
  fields = [
      "sequence_id",
      "paired_sequence_id",
      "episode_id",
      "episode_name",
      "task_id",
      "camera_id",
      "camera_view",
      "manifest_num_frames",
      "actual_num_frames",
      "start_frame",
      "source",
      "scanned_frames",
      "motion_onset_frame",
      "strict_motion_onset_frame",
      "mean_motion_onset_frame",
      "local_baseline_onset_frame",
      "mean_baseline_onset_frame",
      "mean_baseline_threshold",
      "local_baseline_threshold",
      "local_motion_threshold",
      "mean_motion_threshold",
      "status",
      "error",
  ]
  with path.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for row in sorted(audit_rows, key=lambda item: item["sequence_id"]):
      writer.writerow({field: row.get(field, "") for field in fields})


def write_manifest(manifest_path, manifest_rows, start_frames):
  original_fields = list(manifest_rows[0].keys())
  fields = [field for field in original_fields if field != "start_frame"]
  insert_at = fields.index("num_frames") if "num_frames" in fields else len(fields)
  fields = fields[:insert_at] + ["start_frame"] + fields[insert_at:]

  tmp_path = manifest_path.with_suffix(".csv.tmp")
  with tmp_path.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for row in manifest_rows:
      out = {field: row.get(field, "") for field in fields if field != "start_frame"}
      out["start_frame"] = start_frames.get(row["sequence_id"], 0)
      writer.writerow(out)
  os.replace(tmp_path, manifest_path)


def summarize(audit_rows):
  starts = [int(row["start_frame"]) for row in audit_rows]
  nonzero = [value for value in starts if value > 0]
  sources = {}
  for row in audit_rows:
    sources[row["source"]] = sources.get(row["source"], 0) + 1
  errors = sum(row["status"] != "ok" for row in audit_rows)
  if nonzero:
    quantiles = {
        "min": min(nonzero),
        "p50": statistics.median(nonzero),
        "max": max(nonzero),
    }
  else:
    quantiles = {"min": 0, "p50": 0, "max": 0}
  return {
      "human_rows": len(audit_rows),
      "nonzero_start_frames": len(nonzero),
      "zero_start_frames": len(starts) - len(nonzero),
      "errors": errors,
      "quantiles": quantiles,
      "sources": sources,
  }


def main():
  args = parse_args()
  root = Path(args.root)
  manifest_path = root / "manifest.csv"
  audit_path = root / args.audit_name

  with manifest_path.open(newline="") as fp:
    manifest_rows = list(csv.DictReader(fp))
  if not manifest_rows:
    raise RuntimeError(f"Empty manifest: {manifest_path}")

  lookup = read_lookup(root)
  tasks = build_human_tasks(
      root, manifest_rows, lookup, args.max_scan_frames, args.limit)
  print(f"human_tasks={len(tasks)} workers={args.workers} max_scan_frames={args.max_scan_frames}", flush=True)

  audit_rows = []
  started = time.time()
  with Pool(processes=args.workers) as pool:
    for idx, result in enumerate(pool.imap_unordered(process_human, tasks, chunksize=8), 1):
      audit_rows.append(result)
      if idx % 1000 == 0 or idx == len(tasks):
        elapsed = time.time() - started
        rate = idx / elapsed if elapsed else 0.0
        print(f"processed={idx}/{len(tasks)} rate={rate:.1f}/s elapsed={elapsed/60:.1f}m", flush=True)

  start_frames = {row["sequence_id"]: int(row["start_frame"]) for row in audit_rows}
  for row in manifest_rows:
    if row["role"] != "h":
      start_frames[row["sequence_id"]] = 0
    elif row["sequence_id"] not in start_frames:
      start_frames[row["sequence_id"]] = 0

  write_audit(audit_path, audit_rows)

  timestamp = time.strftime("%Y%m%d_%H%M%S")
  backup_path = manifest_path.with_name(f"manifest.before_start_frame_{timestamp}.csv")
  shutil.copy2(manifest_path, backup_path)
  write_manifest(manifest_path, manifest_rows, start_frames)

  summary = summarize(audit_rows)
  print(f"audit={audit_path}", flush=True)
  print(f"backup={backup_path}", flush=True)
  print(f"manifest={manifest_path}", flush=True)
  print(f"summary={summary}", flush=True)


if __name__ == "__main__":
  main()
