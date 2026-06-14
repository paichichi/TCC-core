#!/usr/bin/env python3
"""Build a reusable group-pool index for matched-camera sampling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--index",
      type=Path,
      default=Path("/home/paichichi/data/RH20T/TCC_RH20T/tcn_timestamp_groups.csv"),
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("/home/paichichi/data/RH20T/TCC_RH20T/training_index.pt"),
  )
  return parser.parse_args()


def view_ref(view: dict) -> dict:
  return {
      "rel_path": view["rel_path"],
      "camera_id": int(view["camera_id"]),
  }


def group_record(row: dict, views: list[dict]) -> dict:
  return {
      "timestamp_ms": int(row["target_timestamp_ms"]),
      "views_by_camera": {
          int(view["camera_id"]): view_ref(view)
          for view in views
      },
  }


def main() -> None:
  args = parse_args()
  by_episode_role: dict[tuple[str, str], list[dict]] = defaultdict(list)
  meta: dict[str, dict[str, str]] = {}
  rows = 0

  with args.index.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows += 1
      episode_id = row["episode_id"]
      role = row["role"]
      views = json.loads(row["views_json"])
      meta[episode_id] = {
          "episode_id": episode_id,
          "class_name": row["class_name"],
          "task_id": row["task_id"],
      }
      by_episode_role[(episode_id, role)].append(group_record(row, views))

  episodes = {}
  for episode_id, rec in meta.items():
    h_groups = by_episode_role.get((episode_id, "h"), [])
    r_groups = by_episode_role.get((episode_id, "r"), [])
    if not h_groups or not r_groups:
      continue
    h_groups.sort(key=lambda group: group["timestamp_ms"])
    r_groups.sort(key=lambda group: group["timestamp_ms"])
    episodes[episode_id] = {
        **rec,
        "h_group_pool": h_groups,
        "r_group_pool": r_groups,
    }

  payload = {
      "format": "matched_combo_universal_group_pool_v1",
      "source_index": str(args.index),
      "episodes": episodes,
      "summary": {
          "source_rows": rows,
          "episodes": len(episodes),
          "tasks": len({rec["task_id"] for rec in episodes.values()}),
          "human_groups": sum(len(rec["h_group_pool"]) for rec in episodes.values()),
          "robot_groups": sum(len(rec["r_group_pool"]) for rec in episodes.values()),
      },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, args.output)
  print(args.output)
  print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
  main()
