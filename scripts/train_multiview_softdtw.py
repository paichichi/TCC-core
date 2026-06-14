#!/usr/bin/env python3
"""Train timestamp multi-view fusion with H/R Soft-DTW contrastive loss."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xirl.losses import soft_dtw_sequence_distance  # pylint: disable=wrong-import-position
from xirl.models import ViTB16Backbone  # pylint: disable=wrong-import-position


@dataclass
class ViewRef:
  rel_path: str
  camera_id: int


@dataclass
class Group:
  episode_id: str
  task_id: str
  role: str
  timestamp_ms: int
  views: list[ViewRef]


class FixedSlotFusionSoftDTW(nn.Module):
  """Frozen ViT backbone + fixed camera slot fusion projector."""

  def __init__(
      self,
      num_camera_slots: int,
      embedding_size: int,
      fusion_size: int,
      pretrain_path: str,
      train_layernorm: bool = True,
  ):
    super().__init__()
    self.num_camera_slots = num_camera_slots
    self.backbone = ViTB16Backbone(
        pretrain_path=pretrain_path,
        vit_weights="none",
        pooling="patch_mean",
    )
    for param in self.backbone.parameters():
      param.requires_grad = False
    if train_layernorm:
      for module in self.backbone.modules():
        if isinstance(module, nn.LayerNorm):
          for param in module.parameters():
            param.requires_grad = True

    input_dim = num_camera_slots * self.backbone.output_dim + num_camera_slots
    self.fusion = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, fusion_size),
        nn.GELU(),
        nn.Linear(fusion_size, fusion_size),
        nn.GELU(),
    )
    self.projector = nn.Linear(fusion_size, embedding_size)

  def encode_groups(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
  ) -> torch.Tensor:
    feats = self.backbone(images)
    feats = torch.flatten(feats, 1)
    slots = feats.new_zeros(
        (num_groups, self.num_camera_slots, feats.shape[-1]))
    masks = feats.new_zeros((num_groups, self.num_camera_slots))
    slots[group_indices, camera_ids] = feats
    masks[group_indices, camera_ids] = 1.0
    fused_input = torch.cat([slots.flatten(start_dim=1), masks], dim=-1)
    z = self.projector(self.fusion(fused_input))
    return F.normalize(z, dim=-1)

  def encode_group_subsets(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      subset_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
  ) -> torch.Tensor:
    feats = self.backbone(images)
    feats = torch.flatten(feats, 1)
    slots = feats.new_zeros(
        (num_groups, 2, self.num_camera_slots, feats.shape[-1]))
    masks = feats.new_zeros((num_groups, 2, self.num_camera_slots))
    slots[group_indices, subset_indices, camera_ids] = feats
    masks[group_indices, subset_indices, camera_ids] = 1.0
    fused_input = torch.cat([
        slots.flatten(start_dim=2),
        masks,
    ], dim=-1)
    z = self.projector(self.fusion(fused_input))
    return F.normalize(z, dim=-1)


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
      default=Path("/tmp/tcc-core/multiview_softdtw_runs/smoke"),
  )
  parser.add_argument("--num-timestamps", type=int, default=12)
  parser.add_argument("--batch-pairs", type=int, default=4)
  parser.add_argument("--max-groups", type=int, default=200000)
  parser.add_argument("--max-iters", type=int, default=1000)
  parser.add_argument("--log-every", type=int, default=20)
  parser.add_argument("--save-every", type=int, default=500)
  parser.add_argument("--gamma", type=float, default=0.1)
  parser.add_argument("--temperature", type=float, default=0.1)
  parser.add_argument("--divergence", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--objective",
      choices=["contrastive_softdtw_mv", "paired_softdtw_hr_vvcl"],
      default="contrastive_softdtw_mv",
  )
  parser.add_argument("--lambda-mv", type=float, default=0.0)
  parser.add_argument("--lambda-hr-vvcl", type=float, default=0.5)
  parser.add_argument("--mv-temperature", type=float, default=0.1)
  parser.add_argument("--max-views-per-group", type=int, default=4)
  parser.add_argument("--view-keep-ratio", type=float, default=0.75)
  parser.add_argument("--lr", type=float, default=5e-5)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--embedding-size", type=int, default=128)
  parser.add_argument("--fusion-size", type=int, default=768)
  parser.add_argument("--num-camera-slots", type=int, default=30)
  parser.add_argument(
      "--pretrain-path",
      default="/home/paichichi/data/pretrain/D4R_IN_1M.pth",
  )
  parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
  return parser.parse_args()


def load_tracks(
    index_path: Path,
    min_groups: int,
    max_groups: int | None,
) -> dict[str, dict[str, list[Group]]]:
  tracks: dict[tuple[str, str], list[Group]] = defaultdict(list)
  loaded = 0
  with index_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      views_raw = json.loads(row["views_json"])
      if len(views_raw) < 2:
        continue
      group = Group(
          episode_id=row["episode_id"],
          task_id=row["task_id"],
          role=row["role"],
          timestamp_ms=int(row["target_timestamp_ms"]),
          views=[
              ViewRef(rel_path=view["rel_path"], camera_id=int(view["camera_id"]))
              for view in views_raw
          ],
      )
      tracks[(group.episode_id, group.role)].append(group)
      loaded += 1
      if max_groups is not None and loaded >= max_groups:
        break

  paired: dict[str, dict[str, list[Group]]] = {}
  episode_ids = {episode_id for episode_id, _ in tracks}
  for episode_id in episode_ids:
    h = sorted(tracks.get((episode_id, "h"), []), key=lambda g: g.timestamp_ms)
    r = sorted(tracks.get((episode_id, "r"), []), key=lambda g: g.timestamp_ms)
    if len(h) >= min_groups and len(r) >= min_groups:
      paired[episode_id] = {"h": h, "r": r}
  return paired


def make_transform(image_size: int):
  return transforms.Compose([
      transforms.Resize((image_size, image_size), antialias=True),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225),
      ),
  ])


def stratified_group_sample(
    groups: list[Group],
    num_timestamps: int,
    rng: random.Random,
) -> list[Group]:
  if len(groups) < num_timestamps:
    raise ValueError("Not enough groups to sample.")
  if num_timestamps == 1:
    return [groups[0]]
  selected = [groups[0]]
  middle_count = num_timestamps - 2
  if middle_count > 0:
    middle = groups[1:-1]
    edges = [
        round(i * len(middle) / middle_count)
        for i in range(middle_count + 1)
    ]
    last_local = -1
    for i in range(middle_count):
      lo = max(edges[i], last_local + 1)
      hi = max(lo + 1, edges[i + 1])
      hi = min(hi, len(middle))
      idx = rng.randrange(lo, hi)
      selected.append(middle[idx])
      last_local = idx
  selected.append(groups[-1])
  return selected


def load_image(root: Path, rel_path: str, transform) -> torch.Tensor:
  image = Image.open(root / rel_path).convert("RGB")
  return transform(image)


def sample_view_subset(
    views: list[ViewRef],
    max_views: int,
    keep_ratio: float,
    rng: random.Random,
) -> list[ViewRef]:
  if not views:
    raise ValueError("Cannot sample from an empty view list.")
  selected = list(views)
  rng.shuffle(selected)
  if max_views > 0:
    selected = selected[:max_views]
  keep = max(1, round(len(selected) * keep_ratio))
  keep = min(keep, len(selected))
  return rng.sample(selected, keep)


def make_view_dropout_sequences(
    sequences: list[list[Group]],
    max_views: int,
    keep_ratio: float,
    rng: random.Random,
) -> list[list[tuple[list[ViewRef], list[ViewRef]]]]:
  out = []
  for sequence in sequences:
    subset_sequence = []
    for group in sequence:
      subset_a = sample_view_subset(group.views, max_views, keep_ratio, rng)
      subset_b = sample_view_subset(group.views, max_views, keep_ratio, rng)
      subset_sequence.append((subset_a, subset_b))
    out.append(subset_sequence)
  return out


def prepare_batch_images(
    root: Path,
    sequences: list[list[Group]],
    transform,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
  images = []
  group_indices = []
  camera_ids = []
  group_idx = 0
  for sequence in sequences:
    for group in sequence:
      for view in group.views:
        images.append(load_image(root, view.rel_path, transform))
        group_indices.append(group_idx)
        camera_ids.append(view.camera_id)
      group_idx += 1
  return (
      torch.stack(images, dim=0).to(device),
      torch.tensor(group_indices, dtype=torch.long, device=device),
      torch.tensor(camera_ids, dtype=torch.long, device=device),
      group_idx,
  )


def make_limited_view_sequences(
    sequences: list[list[Group]],
    max_views: int,
    rng: random.Random,
) -> list[list[Group]]:
  out = []
  for sequence in sequences:
    limited_sequence = []
    for group in sequence:
      views = list(group.views)
      rng.shuffle(views)
      if max_views > 0:
        views = views[:max_views]
      limited_sequence.append(Group(
          episode_id=group.episode_id,
          task_id=group.task_id,
          role=group.role,
          timestamp_ms=group.timestamp_ms,
          views=views,
      ))
    out.append(limited_sequence)
  return out


def prepare_subset_batch_images(
    root: Path,
    subset_sequences: list[list[tuple[list[ViewRef], list[ViewRef]]]],
    transform,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
  images = []
  group_indices = []
  subset_indices = []
  camera_ids = []
  group_idx = 0
  for sequence in subset_sequences:
    for subset_a, subset_b in sequence:
      for subset_idx, subset in enumerate([subset_a, subset_b]):
        for view in subset:
          images.append(load_image(root, view.rel_path, transform))
          group_indices.append(group_idx)
          subset_indices.append(subset_idx)
          camera_ids.append(view.camera_id)
      group_idx += 1
  return (
      torch.stack(images, dim=0).to(device),
      torch.tensor(group_indices, dtype=torch.long, device=device),
      torch.tensor(subset_indices, dtype=torch.long, device=device),
      torch.tensor(camera_ids, dtype=torch.long, device=device),
      group_idx,
  )


def compute_softdtw_contrastive(
    h_seq,
    r_seq,
    gamma,
    temperature,
    divergence,
):
  batch_size = h_seq.shape[0]
  h_grid = h_seq[:, None].expand(batch_size, batch_size, *h_seq.shape[1:])
  r_grid = r_seq[None].expand(batch_size, batch_size, *r_seq.shape[1:])
  h_flat = h_grid.reshape(batch_size * batch_size, *h_seq.shape[1:])
  r_flat = r_grid.reshape(batch_size * batch_size, *r_seq.shape[1:])
  distances = soft_dtw_sequence_distance(
      h_flat,
      r_flat,
      gamma=gamma,
      normalize_dimension=False,
      divergence=divergence,
      normalize_time=True,
  ).reshape(batch_size, batch_size)
  labels = torch.arange(batch_size, device=h_seq.device)
  logits = -distances / temperature
  loss = 0.5 * (
      F.cross_entropy(logits, labels) +
      F.cross_entropy(logits.t(), labels)
  )
  top1_hr = (distances.argmin(dim=1) == labels).float().mean()
  top1_rh = (distances.argmin(dim=0) == labels).float().mean()
  pos = distances.diag().mean()
  off = (distances.sum() - distances.diag().sum()) / max(
      1, distances.numel() - batch_size)
  return loss, {
      "distances": distances,
      "top1": 0.5 * (top1_hr + top1_rh),
      "pos_dist": pos,
      "off_dist": off,
  }


def compute_softdtw_paired(
    h_seq,
    r_seq,
    gamma,
    divergence,
):
  distances = soft_dtw_sequence_distance(
      h_seq,
      r_seq,
      gamma=gamma,
      normalize_dimension=False,
      divergence=divergence,
      normalize_time=True,
  )
  return distances.mean(), {
      "distances": distances,
      "top1": torch.ones((), device=h_seq.device),
      "pos_dist": distances.mean(),
      "off_dist": torch.zeros((), device=h_seq.device),
  }


def compute_hr_vvcl(
    h_seq,
    r_seq,
    temperature,
):
  h_global = F.normalize(h_seq.mean(dim=1), dim=-1)
  r_global = F.normalize(r_seq.mean(dim=1), dim=-1)
  logits = (h_global @ r_global.t()) / temperature
  labels = torch.arange(logits.shape[0], device=logits.device)
  loss = 0.5 * (
      F.cross_entropy(logits, labels) +
      F.cross_entropy(logits.t(), labels)
  )
  sims = h_global @ r_global.t()
  top1_hr = (sims.argmax(dim=1) == labels).float().mean()
  top1_rh = (sims.argmax(dim=0) == labels).float().mean()
  diag = sims.diag().mean()
  off = (sims.sum() - sims.diag().sum()) / max(
      1, sims.numel() - logits.shape[0])
  return loss, {
      "top1": 0.5 * (top1_hr + top1_rh),
      "diag": diag,
      "off": off,
  }


def compute_multiview_infonce(
    z_a,
    z_b,
    temperature,
):
  logits = (z_a @ z_b.t()) / temperature
  labels = torch.arange(logits.shape[0], device=logits.device)
  loss = 0.5 * (
      F.cross_entropy(logits, labels) +
      F.cross_entropy(logits.t(), labels)
  )
  sims = z_a @ z_b.t()
  top1_ab = (sims.argmax(dim=1) == labels).float().mean()
  top1_ba = (sims.argmax(dim=0) == labels).float().mean()
  diag = sims.diag().mean()
  off = (sims.sum() - sims.diag().sum()) / max(
      1, sims.numel() - logits.shape[0])
  return loss, {
      "top1": 0.5 * (top1_ab + top1_ba),
      "diag": diag,
      "off": off,
  }


def trainable_summary(model):
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  backbone_total = sum(p.numel() for p in model.backbone.parameters())
  backbone_train = sum(
      p.numel() for p in model.backbone.parameters() if p.requires_grad)
  return (
      f"trainable={trainable}/{total} "
      f"backbone={backbone_train}/{backbone_total}"
  )


def main() -> None:
  args = parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  rng = random.Random(args.seed)
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")

  paired_tracks = load_tracks(args.index, args.num_timestamps, args.max_groups)
  episode_ids = sorted(paired_tracks, key=lambda x: int(x))
  if len(episode_ids) < args.batch_pairs:
    raise RuntimeError("Not enough paired episode tracks.")
  print(
      f"loaded paired_episodes={len(episode_ids)} "
      f"num_timestamps={args.num_timestamps}")

  transform = make_transform(args.image_size)
  model = FixedSlotFusionSoftDTW(
      num_camera_slots=args.num_camera_slots,
      embedding_size=args.embedding_size,
      fusion_size=args.fusion_size,
      pretrain_path=args.pretrain_path,
      train_layernorm=True,
  ).to(device).train()
  print(trainable_summary(model))

  optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad],
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
  scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

  csv_path = args.out_dir / "losses.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step",
        "loss_total",
        "loss_softdtw",
        "loss_aux",
        "loss_aux_h",
        "loss_aux_r",
        "softdtw_top1",
        "softdtw_pos_dist",
        "softdtw_off_dist",
        "aux_top1_h",
        "aux_top1_r",
        "aux_diag_h",
        "aux_off_h",
        "aux_diag_r",
        "aux_off_r",
        "emb_std",
        "batch_pairs",
        "images",
        "seconds",
        "cuda_mem_mb",
    ])
    f.flush()

    start = time.time()
    for step in range(1, args.max_iters + 1):
      iter_start = time.time()
      batch_episode_ids = rng.sample(episode_ids, args.batch_pairs)
      human_sequences = [
          stratified_group_sample(
              paired_tracks[eid]["h"], args.num_timestamps, rng)
          for eid in batch_episode_ids
      ]
      robot_sequences = [
          stratified_group_sample(
              paired_tracks[eid]["r"], args.num_timestamps, rng)
          for eid in batch_episode_ids
      ]
      all_sequences = human_sequences + robot_sequences
      if args.objective == "paired_softdtw_hr_vvcl":
        limited_sequences = make_limited_view_sequences(
            all_sequences, args.max_views_per_group, rng)
        images, group_idx, camera_ids, num_groups = prepare_batch_images(
            args.tcc_root, limited_sequences, transform, device)
      else:
        subset_sequences = make_view_dropout_sequences(
            all_sequences,
            args.max_views_per_group,
            args.view_keep_ratio,
            rng,
        )
        images, group_idx, subset_idx, camera_ids, num_groups = prepare_subset_batch_images(
            args.tcc_root, subset_sequences, transform, device)

      optimizer.zero_grad(set_to_none=True)
      with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
        if args.objective == "paired_softdtw_hr_vvcl":
          z_groups = model.encode_groups(
              images, group_idx, camera_ids, num_groups)
        else:
          z_subsets = model.encode_group_subsets(
              images, group_idx, subset_idx, camera_ids, num_groups)
          z_groups = F.normalize(
              0.5 * (z_subsets[:, 0] + z_subsets[:, 1]), dim=-1)
        z = z_groups.reshape(2 * args.batch_pairs, args.num_timestamps, -1)
        h_seq = z[:args.batch_pairs]
        r_seq = z[args.batch_pairs:]
        if args.objective == "paired_softdtw_hr_vvcl":
          loss_softdtw, metrics = compute_softdtw_paired(
              h_seq,
              r_seq,
              gamma=args.gamma,
              divergence=args.divergence,
          )
          loss_aux, aux = compute_hr_vvcl(
              h_seq, r_seq, args.mv_temperature)
          zero = torch.zeros((), device=device)
          loss_aux_h = loss_aux
          loss_aux_r = zero
          aux_h = aux
          aux_r = {
              "top1": zero,
              "diag": zero,
              "off": zero,
          }
          loss = loss_softdtw + args.lambda_hr_vvcl * loss_aux
        else:
          loss_softdtw, metrics = compute_softdtw_contrastive(
              h_seq,
              r_seq,
              gamma=args.gamma,
              temperature=args.temperature,
              divergence=args.divergence,
          )
          z_sub = z_subsets.reshape(
              2 * args.batch_pairs, args.num_timestamps, 2, -1)
          h_sub = z_sub[:args.batch_pairs].reshape(
              args.batch_pairs * args.num_timestamps, 2, -1)
          r_sub = z_sub[args.batch_pairs:].reshape(
              args.batch_pairs * args.num_timestamps, 2, -1)
          loss_aux_h, aux_h = compute_multiview_infonce(
              h_sub[:, 0], h_sub[:, 1], args.mv_temperature)
          loss_aux_r, aux_r = compute_multiview_infonce(
              r_sub[:, 0], r_sub[:, 1], args.mv_temperature)
          loss_aux = 0.5 * (loss_aux_h + loss_aux_r)
          loss = loss_softdtw + args.lambda_mv * loss_aux
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      seconds = time.time() - iter_start
      mem_mb = (
          torch.cuda.max_memory_allocated(device) / 1024 / 1024
          if device.type == "cuda" else 0.0)
      emb_std = z_groups.detach().float().std(dim=0).mean().item()
      row = [
          step,
          f"{loss.item():.8f}",
          f"{loss_softdtw.item():.8f}",
          f"{loss_aux.item():.8f}",
          f"{loss_aux_h.item():.8f}",
          f"{loss_aux_r.item():.8f}",
          f"{metrics['top1'].item():.6f}",
          f"{metrics['pos_dist'].item():.8f}",
          f"{metrics['off_dist'].item():.8f}",
          f"{aux_h['top1'].item():.6f}",
          f"{aux_r['top1'].item():.6f}",
          f"{aux_h['diag'].item():.6f}",
          f"{aux_h['off'].item():.6f}",
          f"{aux_r['diag'].item():.6f}",
          f"{aux_r['off'].item():.6f}",
          f"{emb_std:.8f}",
          args.batch_pairs,
          images.shape[0],
          f"{seconds:.4f}",
          f"{mem_mb:.1f}",
      ]
      writer.writerow(row)
      if step == 1 or step % args.log_every == 0:
        print(
            f"step {step:05d} total={row[1]} sdtw={row[2]} aux={row[3]} "
            f"sdtw_top1={row[6]} pos/off={row[7]}/{row[8]} "
            f"aux_top1_h/r={row[9]}/{row[10]} "
            f"aux_diag_off_h={row[11]}/{row[12]} "
            f"aux_diag_off_r={row[13]}/{row[14]} "
            f"emb_std={row[15]} pairs={args.batch_pairs} imgs={images.shape[0]} "
            f"mem={row[19]}MB sec={row[18]}",
            flush=True,
        )
        f.flush()
      if args.save_every and step % args.save_every == 0:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            args.out_dir / f"checkpoint_{step:06d}.pt",
        )
    print(f"done in {time.time() - start:.1f}s; losses={csv_path}")


if __name__ == "__main__":
  main()
