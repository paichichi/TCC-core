#!/usr/bin/env python3
"""Audit RH20T training_index image references against a data root."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-root", type=Path, required=True)
  parser.add_argument("--training-index", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--max-print", type=int, default=20)
  return parser.parse_args()


def iter_group_pools(payload: dict):
  for episode_id, rec in payload["episodes"].items():
    for role, pool_name in (("h", "h_group_pool"), ("r", "r_group_pool")):
      for group_idx, group in enumerate(rec.get(pool_name, [])):
        yield episode_id, role, group_idx, group


def group_views(group: dict) -> list[dict]:
  if "views_by_camera" in group:
    return list(group["views_by_camera"].values())
  return list(group.get("views", []))


def set_group_views(group: dict, views: list[dict]) -> None:
  if "views_by_camera" in group:
    group["views_by_camera"] = {
        int(view["camera_id"]): view
        for view in views
    }
  else:
    group["views"] = views


def main() -> None:
  args = parse_args()
  payload = torch.load(args.training_index, map_location="cpu", weights_only=False)
  fmt = payload.get("format")
  if fmt != "matched_combo_universal_group_pool_v1":
    raise ValueError(
        "This audit/repair script currently expects "
        "matched_combo_universal_group_pool_v1."
    )

  checked = 0
  missing: list[tuple[str, str, int, str]] = []
  seen_paths: set[str] = set()
  for episode_id, role, group_idx, group in iter_group_pools(payload):
    for view in group_views(group):
      rel_path = view["rel_path"]
      if rel_path in seen_paths:
        continue
      seen_paths.add(rel_path)
      checked += 1
      if not (args.data_root / rel_path).is_file():
        missing.append((episode_id, role, group_idx, rel_path))

  print(f"format={fmt}")
  print(f"episodes={len(payload['episodes'])}")
  print(f"unique_image_refs={checked}")
  print(f"missing_image_refs={len(missing)}")
  for episode_id, role, group_idx, rel_path in missing[:args.max_print]:
    print(f"missing episode={episode_id} role={role} group={group_idx} path={rel_path}")

  if args.output is None:
    return

  repaired = copy.deepcopy(payload)
  removed_views = 0
  removed_groups = 0
  removed_episodes = []
  for episode_id, rec in list(repaired["episodes"].items()):
    for pool_name in ("h_group_pool", "r_group_pool"):
      kept_groups = []
      for group in rec.get(pool_name, []):
        views = group_views(group)
        kept_views = [
            view for view in views
            if (args.data_root / view["rel_path"]).is_file()
        ]
        removed_views += len(views) - len(kept_views)
        if kept_views:
          new_group = copy.deepcopy(group)
          set_group_views(new_group, kept_views)
          kept_groups.append(new_group)
        else:
          removed_groups += 1
      rec[pool_name] = kept_groups
    if not rec.get("h_group_pool") or not rec.get("r_group_pool"):
      removed_episodes.append(episode_id)
      del repaired["episodes"][episode_id]

  repaired.setdefault("summary", {})
  repaired["summary"]["source_training_index"] = str(args.training_index)
  repaired["summary"]["missing_image_refs_removed"] = len(missing)
  repaired["summary"]["views_removed_by_file_audit"] = removed_views
  repaired["summary"]["groups_removed_by_file_audit"] = removed_groups
  repaired["summary"]["episodes_removed_by_file_audit"] = len(removed_episodes)
  repaired["summary"]["episodes_after_file_audit"] = len(repaired["episodes"])

  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(repaired, args.output)
  print(f"wrote_filtered_index={args.output}")
  print(f"views_removed={removed_views}")
  print(f"groups_removed={removed_groups}")
  print(f"episodes_removed={len(removed_episodes)}")
  print(f"episodes_after={len(repaired['episodes'])}")


if __name__ == "__main__":
  main()
