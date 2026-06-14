#!/usr/bin/env python3
"""Conservative hand-presence boundary trim planning for RH20T human videos.

Default behavior is metadata-only: no video re-encoding and no frame deletion.
If raw source videos are unavailable, the script falls back to the extracted
TCC_RH20T human frame directories. Use --apply-to-frames explicitly to move
head/tail JPG frames out of those extracted directories.
"""

import argparse
import csv
import html
import os
from pathlib import Path
import random
import shutil
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--root",
      default="/home/paichichi/data/RH20T/TCC_RH20T",
      help="TCC_RH20T dataset root containing lookup.csv and debug.csv.",
  )
  parser.add_argument("--num-pairs", type=int, default=50)
  parser.add_argument("--seed", type=int, default=10)
  parser.add_argument(
      "--out-dir",
      default="visualizations/mediapipe_hand_boundary_sample_50",
  )
  parser.add_argument("--padding-sec", type=float, default=2.0)
  parser.add_argument("--stable-window", type=int, default=8)
  parser.add_argument("--stable-fraction", type=float, default=0.5)
  parser.add_argument(
      "--all-hand-fraction",
      type=float,
      default=0.95,
      help=(
          "If hand presence covers at least this fraction of frames, treat the "
          "video as all-hand and fall back to visual static head/tail trimming."
      ),
  )
  parser.add_argument(
      "--visual-diff-threshold",
      type=float,
      default=2.0,
      help="Mean absolute grayscale diff threshold for visual static fallback.",
  )
  parser.add_argument("--min-detection-confidence", type=float, default=0.45)
  parser.add_argument("--max-num-hands", type=int, default=2)
  parser.add_argument(
      "--model-path",
      default="models/hand_landmarker.task",
      help="MediaPipe Hand Landmarker .task model path.",
  )
  parser.add_argument(
      "--model-url",
      default=DEFAULT_MODEL_URL,
      help="URL used to download the model if --model-path is missing.",
  )
  parser.add_argument(
      "--prefer-raw-video",
      action="store_true",
      help="Use raw source_video when it exists; otherwise use extracted JPGs.",
  )
  parser.add_argument(
      "--apply-to-frames",
      action="store_true",
      help="Move extracted JPG frames outside [trim_start, trim_end].",
  )
  return parser.parse_args()


def numeric_jpgs(seq_dir):
  return sorted(seq_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def load_tables(root):
  lookup_path = root / "lookup.csv"
  debug_path = root / "debug.csv"
  with lookup_path.open(newline="") as fp:
    lookup_rows = list(csv.DictReader(fp))
  debug_rows = {}
  with debug_path.open(newline="") as fp:
    for row in csv.DictReader(fp):
      debug_rows[row["sequence_id"]] = row
  return lookup_rows, debug_rows


def select_human_rows(root, num_pairs, seed):
  lookup_rows, debug_rows = load_tables(root)
  rng = random.Random(seed)
  rng.shuffle(lookup_rows)
  selected = []
  for row in lookup_rows:
    seq = row["human_sequence_id"]
    ep = row["class_name"]
    seq_dir = root / "train" / ep / seq
    frames = numeric_jpgs(seq_dir) if seq_dir.is_dir() else []
    debug = debug_rows.get(seq, {})
    source_video = Path(debug.get("source_video", ""))
    if len(frames) < 2 and not source_video.exists():
      continue
    row = dict(row)
    row["_human_dir"] = str(seq_dir)
    row["_source_video"] = str(source_video)
    row["_raw_exists"] = source_video.exists()
    row["_fps"] = float(debug.get("fps") or 25.0)
    row["_frames"] = frames
    selected.append(row)
    if len(selected) == num_pairs:
      break
  if len(selected) < num_pairs:
    raise RuntimeError(f"Only found {len(selected)} usable human sequences.")
  return selected


def frame_from_path(path):
  frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if frame is None:
    raise RuntimeError(f"Failed to read frame: {path}")
  return frame


def iter_extracted_frames(paths):
  for idx, path in enumerate(paths):
    yield idx, frame_from_path(path), path


def iter_video_frames(path):
  cap = cv2.VideoCapture(str(path))
  if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {path}")
  idx = 0
  try:
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      yield idx, frame, None
      idx += 1
  finally:
    cap.release()


def detect_presence(row, hands, prefer_raw_video):
  use_raw = prefer_raw_video and row["_raw_exists"]
  if use_raw:
    frame_iter = iter_video_frames(Path(row["_source_video"]))
    input_kind = "raw_video"
  else:
    frame_iter = iter_extracted_frames(row["_frames"])
    input_kind = "extracted_frames"

  presence = []
  confidences = []
  frame_paths = []
  for idx, frame_bgr, frame_path in frame_iter:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = hands.detect(mp_image)
    has_hand = bool(result.hand_landmarks)
    presence.append(has_hand)
    if result.handedness:
      confidences.append(max(c.score for handed in result.handedness for c in handed))
    else:
      confidences.append(0.0)
    frame_paths.append(str(frame_path) if frame_path is not None else "")
  return input_kind, presence, confidences, frame_paths


def first_stable_true(presence, window, fraction):
  if not presence:
    return None
  if len(presence) < window:
    return 0 if any(presence) else None
  needed = max(1, int(round(window * fraction)))
  for start in range(0, len(presence) - window + 1):
    if sum(presence[start:start + window]) >= needed:
      return start
  return None


def last_stable_true(presence, window, fraction):
  if not presence:
    return None
  if len(presence) < window:
    return len(presence) - 1 if any(presence) else None
  needed = max(1, int(round(window * fraction)))
  for start in range(len(presence) - window, -1, -1):
    if sum(presence[start:start + window]) >= needed:
      return start + window - 1
  return None


def plan_trim(presence, fps, args):
  total = len(presence)
  first = first_stable_true(presence, args.stable_window, args.stable_fraction)
  last = last_stable_true(presence, args.stable_window, args.stable_fraction)
  padding = int(round(args.padding_sec * fps))
  if first is None or last is None or first > last:
    return {
        "boundary_method": "no_hand_keep_full",
        "hand_found": False,
        "hand_presence_ratio": 0.0,
        "visual_head_static_frames": "",
        "visual_tail_static_frames": "",
        "motion_proposal_start_frame": "",
        "motion_proposal_end_frame": "",
        "motion_proposal_note": "",
        "first_stable_hand_frame": "",
        "last_stable_hand_frame": "",
        "trim_start_frame": 0,
        "trim_end_frame": total - 1,
        "removed_head_range": "",
        "removed_tail_range": "",
        "total_frames_after_trim": total,
        "original_final_frame_removed": False,
    }
  trim_start = max(0, first - padding)
  trim_end = min(total - 1, last + padding)
  return {
      "boundary_method": "hand_presence",
      "hand_found": True,
      "hand_presence_ratio": sum(presence) / total if total else 0.0,
      "visual_head_static_frames": "",
      "visual_tail_static_frames": "",
      "motion_proposal_start_frame": "",
      "motion_proposal_end_frame": "",
      "motion_proposal_note": "",
      "first_stable_hand_frame": first,
      "last_stable_hand_frame": last,
      "trim_start_frame": trim_start,
      "trim_end_frame": trim_end,
      "removed_head_range": f"0-{trim_start - 1}" if trim_start > 0 else "",
      "removed_tail_range": f"{trim_end + 1}-{total - 1}" if trim_end < total - 1 else "",
      "total_frames_after_trim": max(0, trim_end - trim_start + 1),
      "original_final_frame_removed": trim_end < total - 1,
  }


def iter_frames_for_row(row, prefer_raw_video):
  use_raw = prefer_raw_video and row["_raw_exists"]
  if use_raw:
    yield from iter_video_frames(Path(row["_source_video"]))
  else:
    yield from iter_extracted_frames(row["_frames"])


def gray_small(frame_bgr):
  gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
  return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def visual_static_plan(row, fps, args):
  grays = []
  for _, frame_bgr, _ in iter_frames_for_row(row, args.prefer_raw_video):
    grays.append(gray_small(frame_bgr))
  total = len(grays)
  if total < 2:
    return {
        "boundary_method": "visual_static_fallback_too_short_keep_full",
        "visual_head_static_frames": "",
        "visual_tail_static_frames": "",
        "motion_proposal_start_frame": "",
        "motion_proposal_end_frame": "",
        "motion_proposal_note": "",
        "trim_start_frame": 0,
        "trim_end_frame": max(0, total - 1),
        "removed_head_range": "",
        "removed_tail_range": "",
        "total_frames_after_trim": total,
        "original_final_frame_removed": False,
    }

  diffs = [
      float(cv2.absdiff(grays[i - 1], grays[i]).mean())
      for i in range(1, total)
  ]
  head_static_edges = 0
  for diff in diffs:
    if diff < args.visual_diff_threshold:
      head_static_edges += 1
    else:
      break
  tail_static_edges = 0
  for diff in reversed(diffs):
    if diff < args.visual_diff_threshold:
      tail_static_edges += 1
    else:
      break

  if head_static_edges == len(diffs) and tail_static_edges == len(diffs):
    return {
        "boundary_method": "visual_static_fallback_all_static_keep_full",
        "visual_head_static_frames": total,
        "visual_tail_static_frames": total,
        "motion_proposal_start_frame": "",
        "motion_proposal_end_frame": "",
        "motion_proposal_note": "all frames are visually static",
        "trim_start_frame": 0,
        "trim_end_frame": total - 1,
        "removed_head_range": "",
        "removed_tail_range": "",
        "total_frames_after_trim": total,
        "original_final_frame_removed": False,
    }

  padding = int(round(args.padding_sec * fps))
  first_motion_frame = head_static_edges + 1
  last_motion_frame = (len(diffs) - tail_static_edges) if tail_static_edges else total - 1
  trim_start = max(0, first_motion_frame - padding)
  trim_end = min(total - 1, last_motion_frame + padding)
  return {
      "boundary_method": "all_hand_visual_static_fallback",
      "visual_head_static_frames": head_static_edges + 1 if head_static_edges else 0,
      "visual_tail_static_frames": tail_static_edges + 1 if tail_static_edges else 0,
      "motion_proposal_start_frame": first_motion_frame,
      "motion_proposal_end_frame": last_motion_frame,
      "motion_proposal_note": "visual static fallback used because hand is present almost everywhere",
      "trim_start_frame": trim_start,
      "trim_end_frame": trim_end,
      "removed_head_range": f"0-{trim_start - 1}" if trim_start > 0 else "",
      "removed_tail_range": f"{trim_end + 1}-{total - 1}" if trim_end < total - 1 else "",
      "total_frames_after_trim": max(0, trim_end - trim_start + 1),
      "original_final_frame_removed": trim_end < total - 1,
  }


def visual_motion_proposal(row, fps, args):
  grays = []
  for _, frame_bgr, _ in iter_frames_for_row(row, args.prefer_raw_video):
    grays.append(gray_small(frame_bgr))
  total = len(grays)
  if total < 2:
    return {
        "motion_proposal_start_frame": "",
        "motion_proposal_end_frame": "",
        "motion_proposal_note": "too few frames for motion proposal",
    }
  diffs = [
      float(cv2.absdiff(grays[i - 1], grays[i]).mean())
      for i in range(1, total)
  ]
  active = [i for i, diff in enumerate(diffs, 1)
            if diff >= args.visual_diff_threshold]
  if not active:
    return {
        "motion_proposal_start_frame": "",
        "motion_proposal_end_frame": "",
        "motion_proposal_note": "no hand detected and no visual motion above threshold",
    }
  padding = int(round(args.padding_sec * fps))
  start = max(0, min(active) - padding)
  end = min(total - 1, max(active) + padding)
  return {
      "motion_proposal_start_frame": start,
      "motion_proposal_end_frame": end,
      "motion_proposal_note": (
          "hand detector found no hands; visual motion suggests possible "
          "interaction interval, but keep_full is enforced"
      ),
  }


def copy_review_frame(src, dst):
  dst.parent.mkdir(parents=True, exist_ok=True)
  if src and Path(src).exists() and not dst.exists():
    shutil.copy2(src, dst)
  return dst


def review_indices(total, trim_start, trim_end):
  candidates = set()
  for center in [0, trim_start, trim_end, total - 1]:
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
      idx = center + offset
      if 0 <= idx < total:
        candidates.add(idx)
  return sorted(candidates)


def write_outputs(out_dir, rows, frame_paths_by_pair, presence_by_pair):
  csv_path = out_dir / "trim_metadata.csv"
  fields = [
      "pair_index",
      "sequence_id",
      "episode",
      "task_id",
      "camera_id",
      "camera_view",
      "input_kind",
      "source_video",
      "source_video_exists",
      "boundary_method",
      "original_total_frames",
      "fps",
      "hand_found",
      "hand_presence_ratio",
      "visual_head_static_frames",
      "visual_tail_static_frames",
      "motion_proposal_start_frame",
      "motion_proposal_end_frame",
      "motion_proposal_note",
      "first_stable_hand_frame",
      "last_stable_hand_frame",
      "trim_start_frame",
      "trim_end_frame",
      "removed_head_range",
      "removed_tail_range",
      "total_frames_after_trim",
      "original_final_frame_removed",
  ]
  with csv_path.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key, "") for key in fields})

  summary_path = out_dir / "summary_by_method.csv"
  summary = {}
  for row in rows:
    method = row["boundary_method"]
    item = summary.setdefault(method, {
        "boundary_method": method,
        "videos": 0,
        "affected": 0,
        "planned_removed_frames": 0,
        "no_hand_keep_full": method == "no_hand_keep_full",
    })
    item["videos"] += 1
    removed = int(row["original_total_frames"]) - int(row["total_frames_after_trim"])
    item["planned_removed_frames"] += removed
    if removed:
      item["affected"] += 1
  with summary_path.open("w", newline="") as fp:
    fields_summary = [
        "boundary_method",
        "videos",
        "affected",
        "planned_removed_frames",
        "no_hand_keep_full",
    ]
    writer = csv.DictWriter(fp, fieldnames=fields_summary)
    writer.writeheader()
    for method in sorted(summary):
      writer.writerow(summary[method])

  sections = []
  for row in rows:
    pair_idx = int(row["pair_index"])
    total = int(row["original_total_frames"])
    trim_start = int(row["trim_start_frame"])
    trim_end = int(row["trim_end_frame"])
    frame_paths = frame_paths_by_pair.get(pair_idx, [])
    presence = presence_by_pair[pair_idx]
    cells = []
    for idx in review_indices(total, trim_start, trim_end):
      is_kept = trim_start <= idx <= trim_end
      has_hand = presence[idx]
      src = frame_paths[idx] if idx < len(frame_paths) else ""
      if src:
        dst = out_dir / "review_frames" / f"pair_{pair_idx:03d}" / f"{idx:06d}.jpg"
        copy_review_frame(src, dst)
        rel = os.path.relpath(dst, out_dir)
        img = f"<img src='{html.escape(rel)}'>"
      else:
        img = "<div class='missing'>raw frame</div>"
      cells.append(
          "<figure class='{} {}'>".format("kept" if is_kept else "removed",
                                          "hand" if has_hand else "nohand") +
          img +
          f"<figcaption>{idx:06d}<br>{'HAND' if has_hand else 'no hand'}<br>{'keep' if is_kept else 'trim'}</figcaption>"
          "</figure>"
      )
    sections.append(
        f"<section><h2>P{pair_idx:02d} H {html.escape(row['sequence_id'])} | "
        f"{html.escape(row['episode'])} | {html.escape(row['task_id'])} | cam {html.escape(row['camera_id'])}</h2>"
        "<div class='meta'>"
        f"<span>Total: <b>{total}</b></span>"
        f"<span>Camera view: <b>{html.escape(row['camera_view'])}</b></span>"
        f"<span>Method: <b>{html.escape(row['boundary_method'])}</b></span>"
        f"<span>Hand ratio: <b>{float(row['hand_presence_ratio']):.3f}</b></span>"
        f"<span>Trim: <b>{trim_start}-{trim_end}</b></span>"
        f"<span>After: <b>{html.escape(str(row['total_frames_after_trim']))}</b></span>"
        f"<span>Head removed: <b>{html.escape(row['removed_head_range'] or '-')}</b></span>"
        f"<span>Tail removed: <b>{html.escape(row['removed_tail_range'] or '-')}</b></span>"
        f"<span>Final removed: <b>{html.escape(str(row['original_final_frame_removed']))}</b></span>"
        f"<span>Hand found: <b>{html.escape(str(row['hand_found']))}</b></span>"
        f"<span>Visual head/tail static: <b>{html.escape(str(row['visual_head_static_frames'] or '-'))} / {html.escape(str(row['visual_tail_static_frames'] or '-'))}</b></span>"
        f"<span>Motion proposal: <b>{html.escape(str(row['motion_proposal_start_frame'] or '-'))}-{html.escape(str(row['motion_proposal_end_frame'] or '-'))}</b></span>"
        "</div><div class='frames'>"
        + "".join(cells) +
        "</div></section>"
    )

  css = """
  body{margin:0;background:#101010;color:#eee;font-family:system-ui,-apple-system,Segoe UI,sans-serif}
  header{position:sticky;top:0;background:#181818;border-bottom:1px solid #333;padding:13px 16px;z-index:2}
  h1{font-size:18px;margin:0 0 6px}.sub{font-size:13px;color:#ccc;margin:0}
  main{padding:16px}section{border-top:1px solid #333;padding:16px 0 22px}h2{font-size:15px;margin:0 0 10px}
  .meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}.meta span{background:#222;border:1px solid #333;border-radius:4px;padding:5px 7px;font-size:12px}.meta b{color:#fff}
  .frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:8px}
  figure{margin:0;border:2px solid #444;border-radius:4px;overflow:hidden;background:#181818}
  figure.kept{border-color:#60a5fa}figure.removed{border-color:#f87171}figure.hand figcaption{color:#86efac}
  img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}.missing{height:74px;display:grid;place-items:center;background:#222;color:#aaa}
  figcaption{text-align:center;font-size:11px;color:#bbb;padding:4px}
  """
  html_path = out_dir / "review.html"
  html_path.write_text(
      "<!doctype html><meta charset='utf-8'><title>MediaPipe hand boundary trim review</title>"
      f"<style>{css}</style><header><h1>MediaPipe Hand Boundary Trim Review</h1>"
      "<p class='sub'>Blue border = kept, red border = would trim. Only head/tail boundaries are planned; middle frames are never removed.</p></header>"
      "<main>" + "".join(sections) + "</main>",
      encoding="utf-8",
  )
  return csv_path, html_path, summary_path


def backup_metadata(root, out_dir):
  for name in ["manifest.csv", "lookup.csv"]:
    src = root / name
    if src.exists():
      shutil.copy2(src, out_dir / f"{name}.before_mediapipe_hand_trim")


def apply_to_extracted_frames(root, out_dir, planned_rows):
  backup_metadata(root, out_dir)
  removed_root = out_dir / "removed_frames"
  sequence_counts = {}
  for row in planned_rows:
    if row["input_kind"] != "extracted_frames":
      continue
    seq_dir = root / "train" / row["episode"] / row["sequence_id"]
    frames = numeric_jpgs(seq_dir)
    keep_start = int(row["trim_start_frame"])
    keep_end = int(row["trim_end_frame"])
    for idx, frame in enumerate(frames):
      if keep_start <= idx <= keep_end:
        continue
      dst = removed_root / row["episode"] / row["sequence_id"] / frame.name
      dst.parent.mkdir(parents=True, exist_ok=True)
      if dst.exists():
        raise FileExistsError(f"Refusing to overwrite backup frame: {dst}")
      shutil.move(str(frame), str(dst))
    sequence_counts[row["sequence_id"]] = len(numeric_jpgs(seq_dir))
  if sequence_counts:
    update_metadata(root, sequence_counts)


def update_metadata(root, sequence_counts):
  manifest_path = root / "manifest.csv"
  tmp = manifest_path.with_suffix(".csv.tmp")
  with manifest_path.open(newline="") as src, tmp.open("w", newline="") as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
      if row["sequence_id"] in sequence_counts:
        row["num_frames"] = str(sequence_counts[row["sequence_id"]])
      writer.writerow(row)
  tmp.replace(manifest_path)

  lookup_path = root / "lookup.csv"
  tmp = lookup_path.with_suffix(".csv.tmp")
  with lookup_path.open(newline="") as src, tmp.open("w", newline="") as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
      seq = row["human_sequence_id"]
      if seq in sequence_counts:
        row["human_num_frames"] = str(sequence_counts[seq])
        try:
          ratio = int(row["robot_num_frames"]) / int(row["human_num_frames"])
          row["robot_over_human_frame_ratio"] = f"{ratio:.4f}"
        except ZeroDivisionError:
          row["robot_over_human_frame_ratio"] = ""
      writer.writerow(row)
  tmp.replace(lookup_path)


def ensure_model(model_path, model_url):
  model_path = Path(model_path)
  if model_path.exists():
    return model_path
  model_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"Downloading MediaPipe model to {model_path} ...")
  urllib.request.urlretrieve(model_url, model_path)
  return model_path


def main():
  args = parse_args()
  root = Path(args.root)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  selected = select_human_rows(root, args.num_pairs, args.seed)
  model_path = ensure_model(args.model_path, args.model_url)

  planned_rows = []
  frame_paths_by_pair = {}
  presence_by_pair = {}
  base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
  options = vision.HandLandmarkerOptions(
      base_options=base_options,
      running_mode=vision.RunningMode.IMAGE,
      num_hands=args.max_num_hands,
      min_hand_detection_confidence=args.min_detection_confidence,
      min_hand_presence_confidence=args.min_detection_confidence,
  )
  with vision.HandLandmarker.create_from_options(options) as hands:
    for pair_idx, row in enumerate(selected, 1):
      input_kind, presence, _, frame_paths = detect_presence(
          row, hands, args.prefer_raw_video)
      plan = plan_trim(presence, row["_fps"], args)
      hand_presence_ratio = sum(presence) / len(presence) if presence else 0.0
      if plan["hand_found"] and hand_presence_ratio >= args.all_hand_fraction:
        visual_plan = visual_static_plan(row, row["_fps"], args)
        plan.update(visual_plan)
        plan["hand_presence_ratio"] = hand_presence_ratio
        plan["hand_found"] = True
      elif not plan["hand_found"]:
        plan.update(visual_motion_proposal(row, row["_fps"], args))
      planned = {
          "pair_index": pair_idx,
          "sequence_id": row["human_sequence_id"],
          "episode": row["class_name"],
          "task_id": row["task_id"],
          "camera_id": row["camera_id"],
          "camera_view": row["camera_view"],
          "input_kind": input_kind,
          "source_video": row["_source_video"],
          "source_video_exists": row["_raw_exists"],
          "original_total_frames": len(presence),
          "fps": row["_fps"],
          **plan,
      }
      planned_rows.append(planned)
      frame_paths_by_pair[pair_idx] = frame_paths
      presence_by_pair[pair_idx] = presence

  csv_path, html_path, summary_path = write_outputs(
      out_dir, planned_rows, frame_paths_by_pair, presence_by_pair)
  if args.apply_to_frames:
    apply_to_extracted_frames(root, out_dir, planned_rows)

  affected = sum(
      bool(row["removed_head_range"] or row["removed_tail_range"])
      for row in planned_rows)
  removed = sum(
      int(row["original_total_frames"]) - int(row["total_frames_after_trim"])
      for row in planned_rows)
  print(f"selected={len(planned_rows)}")
  print(f"affected={affected}")
  print(f"planned_removed_frames={removed}")
  print(f"applied_to_frames={args.apply_to_frames}")
  print(f"metadata={csv_path}")
  print(f"summary={summary_path}")
  print(f"review={html_path}")


if __name__ == "__main__":
  main()
