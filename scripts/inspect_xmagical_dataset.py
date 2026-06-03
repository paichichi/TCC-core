#!/usr/bin/env python3
"""Inspect an XIRL-style X-MAGICAL frame dataset.

Expected layout:
  root/
    train/
      action_class/
        video_dir/
          000000.png
          ...
    valid/
      action_class/
        video_dir/
          000000.png
          ...
"""

from __future__ import annotations

import argparse
import fnmatch
import random
import re
from dataclasses import dataclass
from pathlib import Path


def natural_key(path: Path):
  return [
      int(part) if part.isdigit() else part.lower()
      for part in re.split(r"(\d+)", path.name)
  ]


def sorted_dirs(path: Path, nonempty: bool) -> list[Path]:
  if not path.exists():
    return []
  dirs = [p for p in path.iterdir() if p.is_dir()]
  if nonempty:
    dirs = [p for p in dirs if any(p.iterdir())]
  return sorted(dirs, key=natural_key)


def sorted_frames(path: Path, pattern: str) -> list[Path]:
  return sorted(
      [p for p in path.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, pattern)],
      key=natural_key,
  )


@dataclass
class Video:
  class_name: str
  path: Path
  num_frames: int


def scan_split(split_dir: Path, pattern: str, max_vids_per_class: int) -> dict[str, list[Video]]:
  tree: dict[str, list[Video]] = {}
  for class_dir in sorted_dirs(split_dir, nonempty=True):
    videos = []
    for video_dir in sorted_dirs(class_dir, nonempty=False):
      videos.append(
          Video(
              class_name=class_dir.name,
              path=video_dir,
              num_frames=len(sorted_frames(video_dir, pattern)),
          )
      )
    if max_vids_per_class > 0:
      videos = videos[:max_vids_per_class]
    if videos:
      tree[class_dir.name] = videos
  return tree


def random_batches(tree: dict[str, list[Video]], batch_size: int, seed: int) -> list[list[tuple[str, str]]]:
  rng = random.Random(seed)
  all_idxs = []
  for class_name, videos in tree.items():
    for video in videos:
      all_idxs.append((class_name, video.path.name))
  rng.shuffle(all_idxs)
  if all_idxs and len(all_idxs) < batch_size:
    while len(all_idxs) < batch_size:
      all_idxs.append(rng.choice(all_idxs))
  end = batch_size * (len(all_idxs) // batch_size)
  batches = [all_idxs[i:i + batch_size] for i in range(0, end, batch_size)]
  rng.shuffle(batches)
  return batches


def same_class_batches(tree: dict[str, list[Video]], batch_size: int, seed: int) -> list[list[tuple[str, str]]]:
  rng = random.Random(seed)
  batches = []
  for class_name, videos in tree.items():
    names = [video.path.name for video in videos]
    rng.shuffle(names)
    end = batch_size * (len(names) // batch_size)
    for i in range(0, end, batch_size):
      batches.append([(class_name, name) for name in names[i:i + batch_size]])
  rng.shuffle(batches)
  return batches


def summarize_split(
    split_name: str,
    tree: dict[str, list[Video]],
    batch_size: int,
    num_frames: int,
    num_ctx_frames: int,
    sampler: str,
    seed: int,
    examples: int,
) -> None:
  videos = [video for class_videos in tree.values() for video in class_videos]
  print(f"\n== {split_name} ==")
  if not tree:
    print("missing or empty split")
    return

  total_frames = sum(video.num_frames for video in videos)
  frame_counts = sorted(video.num_frames for video in videos)
  print(f"classes: {len(tree)}")
  print(f"videos: {len(videos)}")
  print(f"frames: {total_frames}")
  print(
      "frames/video: "
      f"min={frame_counts[0]} median={frame_counts[len(frame_counts) // 2]} "
      f"max={frame_counts[-1]}"
  )
  print(
      "expected batch tensor: "
      f"frames shape roughly [{batch_size}, {num_frames * num_ctx_frames}, C, H, W]"
  )

  for class_name, class_videos in tree.items():
    counts = [video.num_frames for video in class_videos]
    empty = sum(1 for count in counts if count == 0)
    print(
        f"- {class_name}: {len(class_videos)} videos, "
        f"frames/video min={min(counts)} max={max(counts)}"
        + (f", empty_videos={empty}" if empty else "")
    )
    for video in class_videos[:examples]:
      print(f"  sample video: {video.path.name} ({video.num_frames} frames)")

  if sampler == "same_class":
    batches = same_class_batches(tree, batch_size, seed)
  else:
    batches = random_batches(tree, batch_size, seed)
  print(f"{sampler} sampler batches/epoch: {len(batches)}")
  for idx, batch in enumerate(batches[:examples]):
    print(f"  batch {idx}: {batch}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default="/tmp/xirl/datasets/xmagical")
  parser.add_argument("--image-ext", default="*.png")
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--num-frames", type=int, default=40)
  parser.add_argument("--num-context-frames", type=int, default=1)
  parser.add_argument("--max-vids-per-class", type=int, default=-1)
  parser.add_argument("--sampler", choices=("random", "same_class"), default="random")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--examples", type=int, default=3)
  args = parser.parse_args()

  root = Path(args.root)
  print(f"root: {root}")
  print(f"exists: {root.exists()}")
  if not root.exists():
    print("Dataset is not present. Run scripts/download_xmagical_dataset.sh first.")
    return

  for split in ("train", "valid"):
    tree = scan_split(root / split, args.image_ext, args.max_vids_per_class)
    summarize_split(
        split,
        tree,
        args.batch_size,
        args.num_frames,
        args.num_context_frames,
        args.sampler,
        args.seed,
        args.examples,
    )


if __name__ == "__main__":
  main()
