#!/usr/bin/env python3
"""Export review folders for matched H/R camera-combo timestamp sampling."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--tcc-root",
      type=Path,
      default=Path("/home/paichichi/data/RH20T/TCC_RH20T"),
  )
  parser.add_argument(
      "--index",
      type=Path,
      default=Path("/home/paichichi/data/RH20T/TCC_RH20T/tcn_timestamp_groups.csv"),
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=Path(
          "/home/paichichi/data/RH20T/TCC_RH20T/reviews/"
          "matched_combo_8timestamps_4view_sample10"
      ),
  )
  parser.add_argument("--num-examples", type=int, default=10)
  parser.add_argument("--num-timestamps", type=int, default=8)
  parser.add_argument("--num-views", type=int, default=4)
  parser.add_argument("--seed", type=int, default=8)
  return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def camera_ids_from_views(views: list[dict]) -> list[int]:
  return sorted({int(view["camera_id"]) for view in views})


def stratified_sample(groups: list[dict], num_timestamps: int, rng: random.Random):
  if len(groups) < num_timestamps:
    raise ValueError(f"need {num_timestamps} groups, got {len(groups)}")
  if num_timestamps == 1:
    return [groups[0]]
  selected = [groups[0]]
  middle_count = num_timestamps - 2
  middle = groups[1:-1]
  edges = [round(i * len(middle) / middle_count) for i in range(middle_count + 1)]
  last_local = -1
  for i in range(middle_count):
    lo = max(edges[i], last_local + 1)
    hi = min(max(lo + 1, edges[i + 1]), len(middle))
    idx = rng.randrange(lo, hi)
    selected.append(middle[idx])
    last_local = idx
  selected.append(groups[-1])
  return selected


def scan_valid_shared_combos(index_path: Path, num_views: int, num_timestamps: int):
  combo_counts: dict[tuple[str, str], Counter[tuple[int, ...]]] = defaultdict(Counter)
  meta: dict[str, dict[str, str]] = {}

  with index_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      episode_id = row["episode_id"]
      role = row["role"]
      meta[episode_id] = {
          "episode_id": episode_id,
          "class_name": row["class_name"],
          "task_id": row["task_id"],
      }
      cams = camera_ids_from_views(json.loads(row["views_json"]))
      if len(cams) < num_views:
        continue
      for combo in itertools.combinations(cams, num_views):
        combo_counts[(episode_id, role)][combo] += 1

  eligible: dict[str, list[tuple[int, ...]]] = {}
  for episode_id in meta:
    h_valid = {
        combo
        for combo, count in combo_counts[(episode_id, "h")].items()
        if count >= num_timestamps
    }
    r_valid = {
        combo
        for combo, count in combo_counts[(episode_id, "r")].items()
        if count >= num_timestamps
    }
    shared = sorted(h_valid & r_valid)
    if shared:
      eligible[episode_id] = shared
  return meta, combo_counts, eligible


def select_examples(
    meta: dict[str, dict[str, str]],
    eligible: dict[str, list[tuple[int, ...]]],
    num_examples: int,
    seed: int,
):
  rng = random.Random(seed)
  episode_ids = sorted(eligible, key=lambda value: int(value))
  rng.shuffle(episode_ids)

  selected = []
  used_tasks = set()
  for episode_id in episode_ids:
    task_id = meta[episode_id]["task_id"]
    if task_id in used_tasks:
      continue
    selected.append(episode_id)
    used_tasks.add(task_id)
    if len(selected) >= num_examples:
      return selected

  for episode_id in episode_ids:
    if episode_id not in selected:
      selected.append(episode_id)
      if len(selected) >= num_examples:
        break
  return selected


def load_selected_groups(
    index_path: Path,
    selected_episodes: set[str],
    episode_combo: dict[str, tuple[int, ...]],
):
  groups_by_er: dict[tuple[str, str], list[dict]] = defaultdict(list)
  with index_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      episode_id = row["episode_id"]
      if episode_id not in selected_episodes:
        continue
      combo = episode_combo[episode_id]
      views = json.loads(row["views_json"])
      by_cam = {int(view["camera_id"]): view for view in views}
      if not all(camera_id in by_cam for camera_id in combo):
        continue
      filtered_views = [by_cam[camera_id] for camera_id in combo]
      groups_by_er[(episode_id, row["role"])].append({
          "group_id": int(row["group_id"]),
          "episode_id": episode_id,
          "class_name": row["class_name"],
          "task_id": row["task_id"],
          "role": row["role"],
          "target_timestamp_ms": int(row["target_timestamp_ms"]),
          "num_views_available": int(row["num_views"]),
          "max_abs_delta_ms": int(row["max_abs_delta_ms"]),
          "views": filtered_views,
      })

  for groups in groups_by_er.values():
    groups.sort(key=lambda group: group["target_timestamp_ms"])
  return groups_by_er


def copy_file(src: Path, dst: Path) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)


def export_native(
    tcc_root: Path,
    side_dir: Path,
    manifest_by_seq: dict[str, dict[str, str]],
    class_name: str,
    selected_groups: list[dict],
):
  camera_to_view: dict[int, dict] = {}
  for group in selected_groups:
    for view in group["views"]:
      camera_to_view[int(view["camera_id"])] = view

  rows = []
  for camera_id in sorted(camera_to_view):
    view = camera_to_view[camera_id]
    sequence_id = view["sequence_id"]
    manifest = manifest_by_seq[sequence_id]
    start_frame = int(manifest.get("start_frame", 0) or 0)
    num_frames = int(manifest["num_frames"])
    cam_dir = side_dir / "native" / f"cam_{camera_id:02d}_{sequence_id}"
    copied = 0
    for frame_idx in range(num_frames):
      src = tcc_root / "train" / class_name / sequence_id / f"{frame_idx:06d}.jpg"
      if not src.exists():
        continue
      copy_file(src, cam_dir / f"{frame_idx:06d}.jpg")
      copied += 1
    rows.append({
        "camera_id": camera_id,
        "camera_view": view["camera_view"],
        "sequence_id": sequence_id,
        "manifest_start_frame": start_frame,
        "manifest_num_frames": num_frames,
        "native_copied_frames": copied,
    })
  return rows


def export_filtered(tcc_root: Path, side_dir: Path, selected_groups: list[dict]):
  rows = []
  for step_idx, group in enumerate(selected_groups):
    for view in group["views"]:
      camera_id = int(view["camera_id"])
      src = tcc_root / view["rel_path"]
      cam_dir = side_dir / "filtered" / f"cam_{camera_id:02d}_{view['sequence_id']}"
      dst = cam_dir / (
          f"step_{step_idx:02d}_group_{group['group_id']:08d}"
          f"_frame_{int(view['frame_idx']):06d}_delta_{int(view['delta_ms']):+04d}.jpg"
      )
      copy_file(src, dst)
      rows.append({
          "step_idx": step_idx,
          "group_id": group["group_id"],
          "target_timestamp_ms": group["target_timestamp_ms"],
          "num_views_available": group["num_views_available"],
          "max_abs_delta_ms": group["max_abs_delta_ms"],
          "camera_id": camera_id,
          "camera_view": view["camera_view"],
          "sequence_id": view["sequence_id"],
          "frame_idx": int(view["frame_idx"]),
          "timestamp_ms": int(view["timestamp_ms"]),
          "delta_ms": int(view["delta_ms"]),
          "rel_path": view["rel_path"],
      })
  return rows


def write_rows(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  args = parse_args()
  if args.out_dir.exists():
    shutil.rmtree(args.out_dir)
  args.out_dir.mkdir(parents=True, exist_ok=True)

  rng = random.Random(args.seed)
  manifest_by_seq = {
      row["sequence_id"]: row
      for row in read_csv(args.tcc_root / "manifest.csv")
  }

  meta, combo_counts, eligible = scan_valid_shared_combos(
      args.index,
      args.num_views,
      args.num_timestamps,
  )
  selected_episodes = select_examples(meta, eligible, args.num_examples, args.seed)
  episode_combo = {
      episode_id: rng.choice(eligible[episode_id])
      for episode_id in selected_episodes
  }
  groups_by_er = load_selected_groups(
      args.index,
      set(selected_episodes),
      episode_combo,
  )

  summary_rows = []
  for example_idx, episode_id in enumerate(selected_episodes, start=1):
    rec = meta[episode_id]
    class_name = rec["class_name"]
    task_id = rec["task_id"]
    combo = episode_combo[episode_id]
    example_dir = args.out_dir / f"P{example_idx:02d}_{class_name}_{task_id}_cams_{'-'.join(map(str, combo))}"

    for role in ["h", "r"]:
      selected_groups = stratified_sample(
          groups_by_er[(episode_id, role)],
          args.num_timestamps,
          rng,
      )
      side_dir = example_dir / ("human" if role == "h" else "robot")
      native_rows = export_native(
          args.tcc_root,
          side_dir,
          manifest_by_seq,
          class_name,
          selected_groups,
      )
      filtered_rows = export_filtered(args.tcc_root, side_dir, selected_groups)
      write_rows(side_dir / "filtered_steps.csv", filtered_rows)
      write_rows(side_dir / "native_tracks.csv", native_rows)

      for native in native_rows:
        matching = [
            row for row in filtered_rows
            if row["camera_id"] == native["camera_id"]
            and row["sequence_id"] == native["sequence_id"]
        ]
        first = matching[0]
        summary_rows.append({
            "example": f"P{example_idx:02d}",
            "episode_id": episode_id,
            "class_name": class_name,
            "task_id": task_id,
            "role": role,
            "shared_camera_combo": "-".join(map(str, combo)),
            "num_valid_shared_combos": len(eligible[episode_id]),
            "h_combo_group_count": combo_counts[(episode_id, "h")][combo],
            "r_combo_group_count": combo_counts[(episode_id, "r")][combo],
            **native,
            "filtered_frames_for_camera": len(matching),
            "filtered_first_group_id": first["group_id"],
            "filtered_first_frame_idx": first["frame_idx"],
        })

  write_rows(args.out_dir / "summary.csv", summary_rows)
  print(args.out_dir)
  print(args.out_dir / "summary.csv")


if __name__ == "__main__":
  main()
