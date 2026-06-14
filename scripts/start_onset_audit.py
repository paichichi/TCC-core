#!/usr/bin/env python3
"""Audit likely task-start onset for RH20T human demos.

This is an onset-screening tool only. It does not trim, move, or delete dataset
frames. The detector is intentionally sensitive to "progress": hand presence,
local hand/shadow motion, or local approach into the task area can all trigger
an onset candidate.
"""

import argparse
import csv
import html
import os
from pathlib import Path
import random
import shutil
import statistics

import cv2

try:
  import mediapipe as mp
  from mediapipe.tasks import python as mp_python
  from mediapipe.tasks.python import vision
except ImportError:
  mp = None
  mp_python = None
  vision = None


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default="/home/paichichi/data/RH20T/TCC_RH20T")
  parser.add_argument("--num-pairs", type=int, default=50)
  parser.add_argument("--seed", type=int, default=10)
  parser.add_argument("--out-dir", default="visualizations/start_onset_audit_50_tuned")
  parser.add_argument("--review-frames", type=int, default=100)
  parser.add_argument("--model-path", default="models/hand_landmarker.task")
  parser.add_argument("--no-mediapipe", action="store_true")
  return parser.parse_args()


def numeric_jpgs(seq_dir):
  return sorted(seq_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def select_human_rows(root, num_pairs, seed):
  with (root / "lookup.csv").open(newline="") as fp:
    rows = list(csv.DictReader(fp))
  rng = random.Random(seed)
  rng.shuffle(rows)
  selected = []
  for row in rows:
    seq_dir = root / "train" / row["class_name"] / row["human_sequence_id"]
    if seq_dir.is_dir() and len(numeric_jpgs(seq_dir)) >= 2:
      selected.append(row)
      if len(selected) == num_pairs:
        break
  if len(selected) < num_pairs:
    raise RuntimeError(f"Only found {len(selected)} usable human sequences.")
  return selected


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


def stable_hand_onset(hand_flags, window=6, min_hits=3):
  return stable_first([1.0 if flag else 0.0 for flag in hand_flags],
                      1.0, window=window, min_hits=min_hits)


def detect_hands(paths, detector):
  if detector is None:
    return [""] * len(paths)
  flags = []
  for path in paths:
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    flags.append(bool(result.hand_landmarks))
  return flags


def build_detector(args):
  if args.no_mediapipe or mp is None:
    return None
  model_path = Path(args.model_path)
  if not model_path.exists():
    return None
  base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
  options = vision.HandLandmarkerOptions(
      base_options=base_options,
      running_mode=vision.RunningMode.IMAGE,
      num_hands=2,
      min_hand_detection_confidence=0.35,
      min_hand_presence_confidence=0.35,
  )
  return vision.HandLandmarker.create_from_options(options)


def onset_for_sequence(paths, hand_flags):
  grays = [small_gray(path) for path in paths]
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

  hand_onset = stable_hand_onset(hand_flags)
  mean_baseline_onset = stable_first(
      base_mean, mean_baseline_threshold, window=4, min_hits=3)
  local_baseline_onset = stable_first(
      base_tile_max, local_baseline_threshold, window=4, min_hits=3)
  strict_local_motion_onset = stable_first(
      adj_tile_max, local_motion_threshold, window=4, min_hits=3)
  sensitive_local_motion_onset = stable_first(
      adj_tile_max, sensitive_local_motion_threshold, window=4, min_hits=2)
  mean_motion_onset = stable_first(
      adj_mean, mean_motion_threshold, window=4, min_hits=2)

  # Strict local motion is used as an automatic progress candidate. Sensitive
  # local motion is reported for review and can support a local-baseline onset.
  motion_onset = sensitive_local_motion_onset
  strict_motion_onset = strict_local_motion_onset

  candidates = []
  sources = []

  def add_candidate(name, value):
    if value != "":
      candidates.append(int(value))
      sources.append((name, int(value)))

  # Motion is the most reliable signal for "a real hand/hand-shadow action has
  # started" in the RH20T overhead views.
  add_candidate("motion", strict_motion_onset)

  # MediaPipe can be excellent when it fires, but is often late or fully blind
  # for tiny edge-of-frame hands. Use it when it agrees with visual progress or
  # when no visual progress exists.
  if hand_onset != "":
    # Ignore immediate-at-start hand detections: in these demos they often
    # correspond to setup clutter or detector false positives, not task onset.
    if int(hand_onset) >= 20 and (
        strict_motion_onset == "" or int(hand_onset) <= int(strict_motion_onset) + 12):
      add_candidate("hand", hand_onset)

  # Local baseline change catches slow approach/shadow cases such as P11/P24.
  # It is intentionally gated: early local changes are often unrelated jiggle,
  # while late local changes usually correspond to a hand entering the task ROI.
  if local_baseline_onset != "":
    local = int(local_baseline_onset)
    mean = int(mean_baseline_onset) if mean_baseline_onset != "" else None
    sensitive_motion = (
        int(sensitive_local_motion_onset)
        if sensitive_local_motion_onset != "" else None)
    strict_motion = (
        int(strict_motion_onset)
        if strict_motion_onset != "" else None)
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

  # Mean baseline is a fallback for large, sustained scene changes.
  if not candidates:
    add_candidate("mean_baseline", mean_baseline_onset)

  progress_onset = min(candidates) if candidates else ""
  progress_sources = [
      name for name, value in sources if progress_onset != "" and value == progress_onset
  ]

  return {
      "progress_onset_frame": progress_onset,
      "progress_onset_source": "+".join(progress_sources),
      "mediapipe_hand_onset_frame": hand_onset,
      "motion_onset_frame": motion_onset,
      "strict_motion_onset_frame": strict_motion_onset,
      "local_motion_onset_frame": sensitive_local_motion_onset,
      "mean_motion_onset_frame": mean_motion_onset,
      "local_baseline_onset_frame": local_baseline_onset,
      "mean_baseline_onset_frame": mean_baseline_onset,
      "mean_baseline_threshold": f"{mean_baseline_threshold:.3f}",
      "local_baseline_threshold": f"{local_baseline_threshold:.3f}",
      "local_motion_threshold": f"{local_motion_threshold:.3f}",
      "mean_motion_threshold": f"{mean_motion_threshold:.3f}",
      "base_mean": base_mean,
      "base_tile_max": base_tile_max,
      "adj_mean": adj_mean,
      "adj_tile_max": adj_tile_max,
  }


def link_or_copy(src, dst):
  dst.parent.mkdir(parents=True, exist_ok=True)
  try:
    os.link(src, dst)
  except OSError:
    shutil.copy2(src, dst)


def write_outputs(out_dir, rows, review_frames):
  report_path = out_dir / "start_onset_audit.csv"
  fields = [
      "pair_index",
      "sequence_id",
      "episode",
      "task_id",
      "camera_id",
      "camera_view",
      "pair_id",
      "total_frames",
      "progress_onset_frame",
      "progress_onset_source",
      "progress_late_ge_30",
      "mediapipe_hand_onset_frame",
      "motion_onset_frame",
      "strict_motion_onset_frame",
      "local_motion_onset_frame",
      "mean_motion_onset_frame",
      "local_baseline_onset_frame",
      "mean_baseline_onset_frame",
      "mean_baseline_threshold",
      "local_baseline_threshold",
      "local_motion_threshold",
      "mean_motion_threshold",
  ]
  with report_path.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in fields})

  nav = []
  sections = []
  for row in rows:
    pair_idx = int(row["pair_index"])
    anchor = f"p{pair_idx:02d}"
    nav.append(f"<a href='#{anchor}'>P{pair_idx:02d}</a>")
    cells = []
    paths = row["_paths"][:review_frames]
    review_dir = out_dir / "review_frames" / f"pair_{pair_idx:03d}_{row['sequence_id']}"
    onset_keys = {
        "progress": row["progress_onset_frame"],
        "hand": row["mediapipe_hand_onset_frame"],
        "motion": row["motion_onset_frame"],
        "local": row["local_baseline_onset_frame"],
        "mean": row["mean_baseline_onset_frame"],
    }
    for idx, src in enumerate(paths):
      dst = review_dir / src.name
      link_or_copy(src, dst)
      rel = os.path.relpath(dst, out_dir)
      classes = [
          name for name, value in onset_keys.items()
          if value != "" and idx == int(value)
      ]
      cells.append(
          f"<figure class='{' '.join(classes)}'>"
          f"<a href='{html.escape(rel)}' target='_blank'>"
          f"<img loading='lazy' src='{html.escape(rel)}'></a>"
          f"<figcaption>{idx:03d}<br>"
          f"bm={row['_base_mean'][idx]:.1f} bt={row['_base_tile_max'][idx]:.1f}<br>"
          f"am={row['_adj_mean'][idx]:.1f} at={row['_adj_tile_max'][idx]:.1f}"
          "</figcaption></figure>"
      )
    sections.append(
        f"<section id='{anchor}'>"
        f"<h2>P{pair_idx:02d} | H {html.escape(row['sequence_id'])} | "
        f"{html.escape(row['episode'])} | {html.escape(row['task_id'])} | "
        f"{html.escape(row['camera_view'])}</h2>"
        "<div class='meta'>"
        f"<span>Progress <b>{html.escape(str(row['progress_onset_frame'] or '-'))}</b> "
        f"({html.escape(row['progress_onset_source'] or '-')})</span>"
        f"<span>Motion <b>{html.escape(str(row['motion_onset_frame'] or '-'))}</b></span>"
        f"<span>Strict motion <b>{html.escape(str(row['strict_motion_onset_frame'] or '-'))}</b></span>"
        f"<span>Hand <b>{html.escape(str(row['mediapipe_hand_onset_frame'] or '-'))}</b></span>"
        f"<span>Local baseline <b>{html.escape(str(row['local_baseline_onset_frame'] or '-'))}</b></span>"
        f"<span>Mean baseline <b>{html.escape(str(row['mean_baseline_onset_frame'] or '-'))}</b></span>"
        f"<span>Total <b>{row['total_frames']}</b></span>"
        "</div>"
        "<div class='legend'>Magenta = final progress onset, cyan = motion, "
        "green = MediaPipe hand, yellow = local baseline, gray = mean baseline. "
        "Captions: bm/base mean, bt/base tile max, am/adj mean, at/adj tile max.</div>"
        "<div class='frames'>" + "".join(cells) + "</div></section>"
    )

  detected = sum(row["progress_onset_frame"] != "" for row in rows)
  late = sum(row["progress_late_ge_30"] is True for row in rows)
  css = """
  :root{color-scheme:dark}body{margin:0;background:#101010;color:#eee;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
  header{position:sticky;top:0;background:#181818;border-bottom:1px solid #333;padding:12px 16px;z-index:5}
  h1{font-size:18px;margin:0 0 6px}.sub{font-size:13px;color:#ccc;margin:0 0 8px}
  .nav{display:flex;gap:6px;flex-wrap:wrap}.nav a{color:#c7d2fe;background:#252525;border:1px solid #3a3a3a;border-radius:4px;padding:3px 7px;text-decoration:none;font-size:12px}
  section{padding:18px 16px 30px;border-top:1px solid #333}h2{font-size:15px;margin:0 0 8px}
  .meta{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 8px}.meta span{background:#202020;border:1px solid #333;border-radius:4px;padding:5px 7px;font-size:12px}
  .legend{font-size:12px;color:#bbb;margin:0 0 8px}.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:7px}
  figure{margin:0;background:#191919;border:1px solid #303030;border-radius:4px;overflow:hidden}
  figure.progress{border:4px solid #f472b6}figure.motion{box-shadow:0 0 0 4px #22d3ee}
  figure.hand{outline:4px solid #22c55e}figure.local{border-color:#facc15}figure.mean figcaption{background:#3f3f46;color:#fff}
  img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#222}figcaption{text-align:center;color:#aaa;font-size:10px;padding:3px 2px}
  """
  html_path = out_dir / "start_onset_review.html"
  html_path.write_text(
      "<!doctype html><meta charset='utf-8'><title>Tuned start onset audit</title>"
      f"<style>{css}</style><header><h1>Tuned Start Onset Audit - 50 Human Demos</h1>"
      f"<p class='sub'>Progress onset detected in {detected}/50; onset >=30 in {late}/50. "
      "No frames are trimmed or deleted.</p>"
      f"<nav class='nav'>{''.join(nav)}</nav></header><main>{''.join(sections)}</main>",
      encoding="utf-8",
  )
  return report_path, html_path


def main():
  args = parse_args()
  root = Path(args.root)
  out_dir = Path(args.out_dir)
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)

  selected = select_human_rows(root, args.num_pairs, args.seed)
  detector = build_detector(args)
  rows = []
  try:
    for pair_idx, row in enumerate(selected, 1):
      seq_dir = root / "train" / row["class_name"] / row["human_sequence_id"]
      paths = numeric_jpgs(seq_dir)
      hand_flags = detect_hands(paths, detector)
      onset = onset_for_sequence(paths, hand_flags)
      progress = onset["progress_onset_frame"]
      out_row = {
          "pair_index": pair_idx,
          "sequence_id": row["human_sequence_id"],
          "episode": row["class_name"],
          "task_id": row["task_id"],
          "camera_id": row["camera_id"],
          "camera_view": row["camera_view"],
          "pair_id": row["pair_id"],
          "total_frames": len(paths),
          "progress_onset_frame": progress,
          "progress_onset_source": onset["progress_onset_source"],
          "progress_late_ge_30": "" if progress == "" else int(progress) >= 30,
          "mediapipe_hand_onset_frame": onset["mediapipe_hand_onset_frame"],
          "motion_onset_frame": onset["motion_onset_frame"],
          "strict_motion_onset_frame": onset["strict_motion_onset_frame"],
          "local_motion_onset_frame": onset["local_motion_onset_frame"],
          "mean_motion_onset_frame": onset["mean_motion_onset_frame"],
          "local_baseline_onset_frame": onset["local_baseline_onset_frame"],
          "mean_baseline_onset_frame": onset["mean_baseline_onset_frame"],
          "mean_baseline_threshold": onset["mean_baseline_threshold"],
          "local_baseline_threshold": onset["local_baseline_threshold"],
          "local_motion_threshold": onset["local_motion_threshold"],
          "mean_motion_threshold": onset["mean_motion_threshold"],
          "_paths": paths,
          "_base_mean": onset["base_mean"],
          "_base_tile_max": onset["base_tile_max"],
          "_adj_mean": onset["adj_mean"],
          "_adj_tile_max": onset["adj_tile_max"],
      }
      rows.append(out_row)
  finally:
    if detector is not None:
      detector.close()

  report_path, html_path = write_outputs(out_dir, rows, args.review_frames)
  detected = sum(row["progress_onset_frame"] != "" for row in rows)
  late = sum(row["progress_late_ge_30"] is True for row in rows)
  print(f"selected={len(rows)}")
  print(f"progress_detected={detected}")
  print(f"progress_late_ge_30={late}")
  print(f"report={report_path}")
  print(f"review={html_path}")


if __name__ == "__main__":
  main()
