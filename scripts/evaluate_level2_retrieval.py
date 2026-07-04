#!/usr/bin/env python3
"""Level-2 H/R retrieval evaluation for trained RH20T checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_multiview_softdtw import (
    FixedSlotFusionSoftDTW,
    build_task_to_episodes,
    compute_softdtw_contrastive,
    load_matched_training_index,
    make_transform,
    materialize_combo_groups,
    prepare_batch_images,
    sample_episode_batch,
    stratified_group_sample,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "checkpoints",
      nargs="+",
      type=Path,
      help="Checkpoint .pt files to evaluate.",
  )
  parser.add_argument(
      "--data-root",
      type=Path,
      default=Path("/home/paichichi/data/RH20T/TCC_RH20T"),
  )
  parser.add_argument(
      "--training-index",
      type=Path,
      default=Path("/home/paichichi/data/RH20T/TCC_RH20T/training_index.pt"),
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=Path("downstream/analysis/level2_retrieval"),
  )
  parser.add_argument("--num-episodes", type=int, default=128)
  parser.add_argument(
      "--sampling",
      choices=["distinct_tasks", "random", "same_task"],
      default="distinct_tasks",
      help=(
          "Episode sampling protocol. same_task is a harder retrieval setting "
          "where all negatives share one task_id."
      ),
  )
  parser.add_argument(
      "--task-id",
      default=None,
      help="Task id used by --sampling same_task. If omitted, uses the largest task.",
  )
  parser.add_argument(
      "--encode-batch-episode-pairs",
      type=int,
      default=8,
      help="Episode pairs encoded per forward pass.",
  )
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--image-size", type=int, default=None)
  parser.add_argument("--gamma", type=float, default=None)
  parser.add_argument("--temperature", type=float, default=None)
  parser.add_argument(
      "--softdtw-divergence",
      action=argparse.BooleanOptionalAction,
      default=None,
  )
  return parser.parse_args()


def checkpoint_run_name(path: Path) -> str:
  if path.parent.name:
    return path.parent.name
  return path.stem


def get_arg(ckpt_args: dict, name: str, default=None):
  return ckpt_args[name] if name in ckpt_args and ckpt_args[name] is not None else default


def load_model(checkpoint_path: Path, device: torch.device):
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  ckpt_args = ckpt.get("args", {})
  model = FixedSlotFusionSoftDTW(
      num_camera_slots=int(get_arg(ckpt_args, "num_camera_slots", 30)),
      embedding_size=int(get_arg(ckpt_args, "embedding_size", 128)),
      fusion_size=int(get_arg(ckpt_args, "fusion_size", 768)),
      pretrain_path="",
      train_layernorm=True,
  )
  model.load_state_dict(ckpt["model"], strict=True)
  return model.to(device).eval(), ckpt_args


def choose_episode_sequences(
    paired_tracks: dict,
    episode_ids: list[str],
    num_timestamps: int,
    rng: random.Random,
):
  human_sequences = []
  robot_sequences = []
  chosen_combo_ids = []
  for episode_id in episode_ids:
    combo_rec = rng.choice(paired_tracks[episode_id]["shared_combos"])
    combo = combo_rec["camera_ids"]
    h_groups = materialize_combo_groups(
        paired_tracks[episode_id]["h_group_pool"],
        combo_rec["h_group_indices"],
        combo,
        episode_id,
        paired_tracks[episode_id]["task_id"],
        "h",
    )
    r_groups = materialize_combo_groups(
        paired_tracks[episode_id]["r_group_pool"],
        combo_rec["r_group_indices"],
        combo,
        episode_id,
        paired_tracks[episode_id]["task_id"],
        "r",
    )
    human_sequences.append(stratified_group_sample(h_groups, num_timestamps, rng))
    robot_sequences.append(stratified_group_sample(r_groups, num_timestamps, rng))
    chosen_combo_ids.append("-".join(str(camera_id) for camera_id in combo))
  return human_sequences, robot_sequences, chosen_combo_ids


@torch.no_grad()
def encode_sequences(
    model,
    data_root: Path,
    transform,
    device: torch.device,
    human_sequences,
    robot_sequences,
    encode_batch_episode_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  h_chunks = []
  r_chunks = []
  num_timestamps = len(human_sequences[0])
  for start in range(0, len(human_sequences), encode_batch_episode_pairs):
    end = min(start + encode_batch_episode_pairs, len(human_sequences))
    chunk_h = human_sequences[start:end]
    chunk_r = robot_sequences[start:end]
    sequences = chunk_h + chunk_r
    images, group_idx, camera_ids, num_groups = prepare_batch_images(
        data_root, sequences, transform, device)
    z_groups, _ = model.encode_groups(images, group_idx, camera_ids, num_groups)
    z_groups = z_groups.float().cpu()
    chunk_size = end - start
    z = z_groups.reshape(2 * chunk_size, num_timestamps, -1)
    h_chunks.append(z[:chunk_size])
    r_chunks.append(z[chunk_size:])
    del images, group_idx, camera_ids, z_groups, z
    if device.type == "cuda":
      torch.cuda.empty_cache()
  return torch.cat(h_chunks, dim=0), torch.cat(r_chunks, dim=0)


def rank_metrics(distances: torch.Tensor) -> dict[str, float]:
  n = distances.shape[0]
  labels = torch.arange(n)
  h_order = distances.argsort(dim=1)
  r_order = distances.argsort(dim=0)
  h_ranks = (h_order == labels[:, None]).nonzero()[:, 1] + 1
  r_ranks = (r_order == labels[None, :]).nonzero()[:, 0] + 1

  def summarize(prefix: str, ranks: torch.Tensor) -> dict[str, float]:
    ranks_f = ranks.float()
    return {
        f"{prefix}_top1": (ranks <= 1).float().mean().item(),
        f"{prefix}_top5": (ranks <= min(5, n)).float().mean().item(),
        f"{prefix}_top10": (ranks <= min(10, n)).float().mean().item(),
        f"{prefix}_mean_rank": ranks_f.mean().item(),
        f"{prefix}_median_rank": ranks_f.median().item(),
        f"{prefix}_mrr": (1.0 / ranks_f).mean().item(),
    }

  out = {}
  out.update(summarize("h2r", h_ranks))
  out.update(summarize("r2h", r_ranks))
  diag = distances.diag()
  off_mask = ~torch.eye(n, dtype=torch.bool)
  off = distances[off_mask]
  out["pos_dist"] = diag.mean().item()
  out["off_dist"] = off.mean().item()
  out["margin"] = (off.mean() - diag.mean()).item()
  return out


def write_summary(path: Path, rows: list[dict]) -> None:
  if not rows:
    return
  fields = []
  for row in rows:
    for key in row:
      if key not in fields:
        fields.append(key)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def write_distance_matrix(path: Path, episode_ids: list[str], distances: torch.Tensor):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["human_episode_id", *episode_ids])
    for episode_id, row in zip(episode_ids, distances.tolist()):
      writer.writerow([episode_id, *[f"{value:.8f}" for value in row]])


def select_episode_ids(
    paired_tracks: dict,
    task_to_episodes: dict[str, list[str]],
    all_episode_ids: list[str],
    num_episodes: int,
    sampling: str,
    task_id: str | None,
    rng: random.Random,
) -> tuple[list[str], str | None]:
  if sampling == "distinct_tasks":
    return (
        sample_episode_batch(
            all_episode_ids,
            task_to_episodes,
            min(num_episodes, len(all_episode_ids)),
            prefer_distinct_tasks=True,
            rng=rng,
        ),
        None,
    )
  if sampling == "random":
    count = min(num_episodes, len(all_episode_ids))
    return rng.sample(all_episode_ids, count), None

  if task_id is None:
    task_id = max(task_to_episodes, key=lambda key: len(task_to_episodes[key]))
  candidates = sorted(task_to_episodes[task_id], key=lambda value: int(value))
  if len(candidates) < num_episodes:
    print(
        f"task {task_id} only has {len(candidates)} eligible episodes; "
        f"using all of them.",
        flush=True,
    )
  count = min(num_episodes, len(candidates))
  return rng.sample(candidates, count), task_id


def main() -> None:
  args = parse_args()
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  summary_rows = []

  for checkpoint_path in args.checkpoints:
    run_name = checkpoint_run_name(checkpoint_path)
    print(f"evaluating {run_name}", flush=True)
    model, ckpt_args = load_model(checkpoint_path, device)
    num_timestamps = int(get_arg(ckpt_args, "num_timestamps"))
    num_multi_view = int(get_arg(ckpt_args, "num_multi_view"))
    image_size = int(args.image_size or get_arg(ckpt_args, "image_size", 224))
    gamma = float(args.gamma if args.gamma is not None else get_arg(ckpt_args, "gamma", 0.1))
    temperature = float(
        args.temperature
        if args.temperature is not None
        else get_arg(ckpt_args, "temperature", 0.1))
    softdtw_divergence = (
        bool(args.softdtw_divergence)
        if args.softdtw_divergence is not None
        else bool(get_arg(ckpt_args, "softdtw_divergence", True))
    )

    paired_tracks = load_matched_training_index(
        args.training_index,
        min_groups=num_timestamps,
        combo_size=num_multi_view,
    )
    episode_ids = sorted(paired_tracks, key=lambda value: int(value))
    task_to_episodes = build_task_to_episodes(paired_tracks)
    rng = random.Random(args.seed)
    selected_episode_ids, selected_task_id = select_episode_ids(
        paired_tracks,
        task_to_episodes,
        episode_ids,
        args.num_episodes,
        args.sampling,
        args.task_id,
        rng,
    )
    human_sequences, robot_sequences, combo_ids = choose_episode_sequences(
        paired_tracks, selected_episode_ids, num_timestamps, rng)
    transform = make_transform(image_size)
    h_seq, r_seq = encode_sequences(
        model,
        args.data_root,
        transform,
        device,
        human_sequences,
        robot_sequences,
        args.encode_batch_episode_pairs,
    )
    _, metrics = compute_softdtw_contrastive(
        h_seq.to(device),
        r_seq.to(device),
        gamma=gamma,
        temperature=temperature,
        softdtw_divergence=softdtw_divergence,
    )
    distances = metrics["distances"].detach().cpu().float()
    retrieval = rank_metrics(distances)
    row = {
        "run": run_name,
        "checkpoint": str(checkpoint_path),
        "num_episodes": len(selected_episode_ids),
        "num_timestamps": num_timestamps,
        "num_multi_view": num_multi_view,
        "image_size": image_size,
        "gamma": gamma,
        "temperature": temperature,
        "softdtw_divergence": softdtw_divergence,
        "sampling": args.sampling,
        "task_id": selected_task_id or "",
        **retrieval,
    }
    summary_rows.append(row)

    run_out = args.out_dir / run_name
    write_distance_matrix(run_out / "distance_matrix.csv", selected_episode_ids, distances)
    write_summary(run_out / "episode_sample.csv", [
        {
            "episode_id": episode_id,
            "camera_combo": combo_id,
        }
        for episode_id, combo_id in zip(selected_episode_ids, combo_ids)
    ])
    print(
        f"{run_name}: h2r_top1={retrieval['h2r_top1']:.4f} "
        f"r2h_top1={retrieval['r2h_top1']:.4f} "
        f"margin={retrieval['margin']:.4f}",
        flush=True,
    )
    del model
    if device.type == "cuda":
      torch.cuda.empty_cache()

  write_summary(args.out_dir / "retrieval_summary.csv", summary_rows)
  print(f"wrote: {args.out_dir / 'retrieval_summary.csv'}")


if __name__ == "__main__":
  main()
