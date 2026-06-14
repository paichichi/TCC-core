#!/usr/bin/env python3
"""Sample human-robot pairs and trim static human tail frames.

This is intentionally dependency-light: frame decoding is delegated to ffmpeg
and the rest uses the Python standard library.
"""

import argparse
import csv
import html
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--root",
      default="/home/paichichi/data/RH20T/TCC_RH20T",
      help="TCC_RH20T dataset root.",
  )
  parser.add_argument("--num-pairs", type=int, default=50)
  parser.add_argument("--seed", type=int, default=8)
  parser.add_argument("--out-dir", default="visualizations/static_tail_sample_50")
  parser.add_argument("--tail-scan", type=int, default=120)
  parser.add_argument("--diff-threshold", type=float, default=2.0)
  parser.add_argument("--min-static-tail", type=int, default=10)
  parser.add_argument("--keep-static", type=int, default=5)
  parser.add_argument(
      "--min-after-frames",
      type=int,
      default=16,
      help="Never trim a sequence below this many remaining frames.",
  )
  parser.add_argument(
      "--remove-from-final",
      action="store_true",
      help=(
          "Remove static tail frames starting at the original final frame. "
          "By default, the script keeps the final keep-static frames as a "
          "goal-state anchor and removes earlier frames inside the static tail."
      ),
  )
  parser.add_argument(
      "--trim-roles",
      default="h",
      help="Comma-separated roles to trim. Default trims human only.",
  )
  parser.add_argument(
      "--apply",
      action="store_true",
      help="Move redundant frames out of the dataset and update metadata.",
  )
  return parser.parse_args()


def numeric_jpgs(seq_dir):
  return sorted(seq_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def decode_gray_frames(paths, width=64, height=36):
  if not paths:
    return []
  frame_size = width * height
  with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fp:
    list_path = Path(fp.name)
    for path in paths:
      escaped = str(path).replace("'", "'\\''")
      fp.write(f"file '{escaped}'\n")
  try:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-safe",
        "0",
        "-f",
        "concat",
        "-i",
        str(list_path),
        "-vf",
        f"scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
  finally:
    list_path.unlink(missing_ok=True)

  data = result.stdout
  usable = len(data) // frame_size
  return [
      data[i * frame_size:(i + 1) * frame_size]
      for i in range(usable)
  ]


def mean_abs_diff(a, b):
  return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def inspect_tail(frames, args):
  scan_frames = frames[-args.tail_scan:]
  decoded = decode_gray_frames(scan_frames)
  if len(decoded) < 2:
    return {
        "static_tail_len": len(decoded),
        "remove_count": 0,
        "diffs": [],
        "remove_paths": [],
        "after_frames": list(frames),
    }

  diffs = [mean_abs_diff(decoded[i - 1], decoded[i])
           for i in range(1, len(decoded))]
  trailing_static_edges = 0
  for diff in reversed(diffs):
    if diff < args.diff_threshold:
      trailing_static_edges += 1
    else:
      break
  static_tail_len = trailing_static_edges + 1
  remove_count = 0
  remove_paths = []
  if static_tail_len >= args.min_static_tail and static_tail_len > args.keep_static:
    if args.remove_from_final:
      remove_count = min(
          static_tail_len,
          max(0, len(frames) - args.min_after_frames),
      )
      start = len(frames) - remove_count
      end = len(frames)
    else:
      remove_count = min(
          static_tail_len - args.keep_static,
          max(0, len(frames) - args.min_after_frames),
      )
      start = len(frames) - static_tail_len
      end = start + remove_count
    remove_paths = frames[start:end]
  remove_set = set(remove_paths)
  after_frames = [path for path in frames if path not in remove_set]
  return {
      "static_tail_len": static_tail_len,
      "remove_count": remove_count,
      "diffs": diffs,
      "remove_paths": remove_paths,
      "after_frames": after_frames,
  }


def select_pairs(root, num_pairs, seed):
  lookup_path = root / "lookup.csv"
  with lookup_path.open(newline="") as fp:
    rows = list(csv.DictReader(fp))
  rng = random.Random(seed)
  rng.shuffle(rows)
  selected = []
  for row in rows:
    ep = f"episode_{int(row['episode_id']):06d}"
    human_dir = root / "train" / ep / row["human_sequence_id"]
    robot_dir = root / "train" / ep / row["robot_sequence_id"]
    if not human_dir.is_dir() or not robot_dir.is_dir():
      continue
    human_frames = numeric_jpgs(human_dir)
    robot_frames = numeric_jpgs(robot_dir)
    if len(human_frames) < 10 or len(robot_frames) < 10:
      continue
    row = dict(row)
    row["_episode_dir"] = ep
    row["_human_dir"] = str(human_dir)
    row["_robot_dir"] = str(robot_dir)
    row["_human_frames"] = human_frames
    row["_robot_frames"] = robot_frames
    selected.append(row)
    if len(selected) == num_pairs:
      break
  if len(selected) < num_pairs:
    raise RuntimeError(f"Only found {len(selected)} valid pairs.")
  return selected


def copy_preview_images(out_dir, pair_idx, role, label, paths):
  preview_dir = out_dir / "preview_frames" / f"pair_{pair_idx:03d}" / f"{role}_{label}"
  preview_dir.mkdir(parents=True, exist_ok=True)
  copied = []
  for i, path in enumerate(paths, 1):
    dst = preview_dir / f"{i:02d}_{path.name}"
    if not dst.exists():
      shutil.copy2(path, dst)
    copied.append(dst)
  return copied


def write_report(out_dir, reports):
  report_path = out_dir / "report.csv"
  fields = [
      "pair_index",
      "class_name",
      "task_id",
      "camera_id",
      "role",
      "sequence_id",
      "before_frames",
      "static_tail_len",
      "remove_count",
      "after_frames",
      "trim_applied",
      "first_removed",
      "last_removed",
  ]
  with report_path.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    for row in reports:
      writer.writerow({field: row.get(field, "") for field in fields})
  return report_path


def write_html(out_dir, selected, plans, args):
  html_path = out_dir / "preview.html"
  css = """
  body{font-family:system-ui,sans-serif;margin:18px;background:#111;color:#eee}
  .pair{margin:0 0 34px}.row{display:grid;grid-template-columns:132px repeat(10,128px);gap:5px;align-items:center;margin:5px 0}
  .label{font-size:12px;line-height:1.35;color:#ddd}.thumb{width:128px;height:72px;object-fit:cover;background:#222}
  .cap{font-size:10px;color:#aaa;text-align:center}.cell{display:flex;flex-direction:column;gap:2px}
  h1{font-size:18px}h2{font-size:15px;margin:0 0 8px}code{color:#ddd}.trim{color:#7dd3fc}.skip{color:#bbb}
  """
  parts = [
      "<!doctype html><meta charset='utf-8'>",
      "<title>Static tail sample preview</title>",
      f"<style>{css}</style>",
      "<h1>Static Tail Sample Preview</h1>",
      (
          f"<p>{args.num_pairs} random pairs, threshold={args.diff_threshold}, "
          f"min_static_tail={args.min_static_tail}, keep_static={args.keep_static}, "
          f"apply={args.apply}, trim_roles={html.escape(args.trim_roles)}</p>"
      ),
  ]
  for pair_idx, row in enumerate(selected, 1):
    parts.append(
        f"<section class='pair'><h2>Pair {pair_idx}: "
        f"{html.escape(row['class_name'])} / {html.escape(row['task_id'])} / "
        f"camera {html.escape(row['camera_id'])} / {html.escape(row['embodiment'])}</h2>"
    )
    for role_name, role_key, seq_key, frame_key in [
        ("human", "h", "human_sequence_id", "_human_frames"),
        ("robot", "r", "robot_sequence_id", "_robot_frames"),
    ]:
      plan = plans[(pair_idx, role_key)]
      before_paths = copy_preview_images(
          out_dir, pair_idx, role_key, "before", row[frame_key][-10:])
      after_paths = copy_preview_images(
          out_dir, pair_idx, role_key, "after", plan["after_frames"][-10:])
      tag_class = "trim" if plan["remove_count"] else "skip"
      status = (
          f"remove {plan['remove_count']}"
          if plan["remove_count"] else "no trim"
      )
      for label, paths in [("before", before_paths), ("after", after_paths)]:
        parts.append(
            f"<div class='row'><div class='label'><b>{role_name} {label}</b><br>"
            f"<code>{html.escape(row[seq_key])}</code><br>"
            f"static tail {plan['static_tail_len']}<br>"
            f"<span class='{tag_class}'>{status}</span></div>"
        )
        for path in paths:
          rel = os.path.relpath(path, out_dir)
          parts.append(
              f"<div class='cell'><img class='thumb' src='{html.escape(rel)}'>"
              f"<div class='cap'>{html.escape(path.stem[-10:])}</div></div>"
          )
        parts.append("</div>")
    parts.append("</section>")
  html_path.write_text("\n".join(parts), encoding="utf-8")
  return html_path


def backup_metadata(root, out_dir):
  for name in ["manifest.csv", "lookup.csv"]:
    src = root / name
    if src.exists():
      shutil.copy2(src, out_dir / f"{name}.before_static_tail_sample")


def update_metadata(root, sequence_counts, pair_counts):
  manifest_path = root / "manifest.csv"
  tmp_manifest = manifest_path.with_suffix(".csv.tmp")
  with manifest_path.open(newline="") as src, tmp_manifest.open("w", newline="") as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
      if row["sequence_id"] in sequence_counts:
        row["num_frames"] = str(sequence_counts[row["sequence_id"]])
      writer.writerow(row)
  tmp_manifest.replace(manifest_path)

  lookup_path = root / "lookup.csv"
  tmp_lookup = lookup_path.with_suffix(".csv.tmp")
  with lookup_path.open(newline="") as src, tmp_lookup.open("w", newline="") as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
      key = (row["class_name"], row["human_sequence_id"], row["robot_sequence_id"])
      if key in pair_counts:
        counts = pair_counts[key]
        if "h" in counts:
          row["human_num_frames"] = str(counts["h"])
        if "r" in counts:
          row["robot_num_frames"] = str(counts["r"])
        try:
          ratio = int(row["robot_num_frames"]) / int(row["human_num_frames"])
          row["robot_over_human_frame_ratio"] = f"{ratio:.4f}"
        except ZeroDivisionError:
          row["robot_over_human_frame_ratio"] = ""
      writer.writerow(row)
  tmp_lookup.replace(lookup_path)


def apply_trims(out_dir, selected, plans, trim_roles):
  removed_root = out_dir / "removed_frames"
  sequence_counts = {}
  pair_counts = {}
  for pair_idx, row in enumerate(selected, 1):
    pair_key = (row["class_name"], row["human_sequence_id"], row["robot_sequence_id"])
    pair_counts.setdefault(pair_key, {})
    for role_key, seq_key, dir_key in [
        ("h", "human_sequence_id", "_human_dir"),
        ("r", "robot_sequence_id", "_robot_dir"),
    ]:
      plan = plans[(pair_idx, role_key)]
      if role_key not in trim_roles or not plan["remove_paths"]:
        continue
      dst_dir = removed_root / row["_episode_dir"] / row[seq_key]
      dst_dir.mkdir(parents=True, exist_ok=True)
      for src in plan["remove_paths"]:
        dst = dst_dir / src.name
        if dst.exists():
          raise FileExistsError(f"Refusing to overwrite backup frame: {dst}")
        shutil.move(str(src), str(dst))
      new_count = len(numeric_jpgs(Path(row[dir_key])))
      sequence_counts[row[seq_key]] = new_count
      pair_counts[pair_key][role_key] = new_count
  return sequence_counts, pair_counts


def main():
  args = parse_args()
  root = Path(args.root)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  trim_roles = {role.strip() for role in args.trim_roles.split(",") if role.strip()}

  selected = select_pairs(root, args.num_pairs, args.seed)
  plans = {}
  reports = []
  for pair_idx, row in enumerate(selected, 1):
    for role_key, seq_key, frame_key in [
        ("h", "human_sequence_id", "_human_frames"),
        ("r", "robot_sequence_id", "_robot_frames"),
    ]:
      frames = row[frame_key]
      plan = inspect_tail(frames, args)
      if role_key not in trim_roles:
        plan["remove_paths"] = []
        plan["remove_count"] = 0
        plan["after_frames"] = list(frames)
      plans[(pair_idx, role_key)] = plan
      reports.append({
          "pair_index": pair_idx,
          "class_name": row["class_name"],
          "task_id": row["task_id"],
          "camera_id": row["camera_id"],
          "role": role_key,
          "sequence_id": row[seq_key],
          "before_frames": len(frames),
          "static_tail_len": plan["static_tail_len"],
          "remove_count": plan["remove_count"],
          "after_frames": len(plan["after_frames"]),
          "trim_applied": bool(args.apply and plan["remove_count"]),
          "first_removed": plan["remove_paths"][0].name if plan["remove_paths"] else "",
          "last_removed": plan["remove_paths"][-1].name if plan["remove_paths"] else "",
      })

  html_path = write_html(out_dir, selected, plans, args)
  report_path = write_report(out_dir, reports)

  if args.apply:
    backup_metadata(root, out_dir)
    sequence_counts, pair_counts = apply_trims(out_dir, selected, plans, trim_roles)
    update_metadata(root, sequence_counts, pair_counts)

  removed_total = sum(row["remove_count"] for row in reports)
  affected = sum(1 for row in reports if row["remove_count"])
  print(f"selected_pairs={len(selected)}")
  print(f"affected_sequences={affected}")
  print(f"planned_removed_frames={removed_total}")
  print(f"applied={args.apply}")
  print(f"report={report_path}")
  print(f"preview={html_path}")
  if args.apply:
    print(f"removed_frames={out_dir / 'removed_frames'}")


if __name__ == "__main__":
  try:
    main()
  except subprocess.CalledProcessError as exc:
    print(f"Command failed: {exc}", file=sys.stderr)
    sys.exit(exc.returncode)
