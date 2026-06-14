#!/usr/bin/env python3
"""Train timestamp multi-view VVCL on RH20T timestamp groups."""

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

from xirl.models import ViTB16Backbone  # pylint: disable=wrong-import-position


@dataclass
class ViewRef:
  rel_path: str
  camera_id: int


@dataclass
class Group:
  episode_id: str
  role: str
  timestamp_ms: int
  views: list[ViewRef]


class FixedSlotMultiViewVVCL(nn.Module):
  """Frozen ViT backbone + fixed camera slot fusion + SigLIP projector."""

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
    self.logit_scale = nn.Parameter(torch.tensor(1.0))
    self.logit_bias = nn.Parameter(torch.tensor(0.0))

  def encode_subsets(
      self,
      images: torch.Tensor,
      pair_indices: torch.Tensor,
      subset_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_pairs: int,
  ) -> torch.Tensor:
    feats = self.backbone(images)
    feats = torch.flatten(feats, 1)
    slots = feats.new_zeros(
        (num_pairs, 2, self.num_camera_slots, feats.shape[-1]))
    masks = feats.new_zeros((num_pairs, 2, self.num_camera_slots))
    slots[pair_indices, subset_indices, camera_ids] = feats
    masks[pair_indices, subset_indices, camera_ids] = 1.0

    flat = torch.cat([
        slots.flatten(start_dim=2),
        masks,
    ], dim=-1)
    z = self.projector(self.fusion(flat))
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
      default=Path("/tmp/tcc-core/multiview_vvcl_runs/smoke"),
  )
  parser.add_argument("--num-timestamps", type=int, default=12)
  parser.add_argument("--episodes-per-role", type=int, default=1)
  parser.add_argument("--max-groups", type=int, default=200000)
  parser.add_argument("--max-iters", type=int, default=1000)
  parser.add_argument("--log-every", type=int, default=20)
  parser.add_argument("--save-every", type=int, default=500)
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


def load_groups(
    index_path: Path,
    min_groups: int,
    max_groups: int | None,
) -> dict[str, list[list[Group]]]:
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
          role=row["role"],
          timestamp_ms=int(row["target_timestamp_ms"]),
          views=[
              ViewRef(
                  rel_path=view["rel_path"],
                  camera_id=int(view["camera_id"]),
              )
              for view in views_raw
          ],
      )
      tracks[(group.episode_id, group.role)].append(group)
      loaded += 1
      if max_groups is not None and loaded >= max_groups:
        break

  by_role: dict[str, list[list[Group]]] = {"h": [], "r": []}
  for (_, role), groups in tracks.items():
    groups.sort(key=lambda group: group.timestamp_ms)
    if len(groups) >= min_groups:
      by_role[role].append(groups)
  return by_role


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
  edges = [
      round(i * len(groups) / num_timestamps)
      for i in range(num_timestamps + 1)
  ]
  selected = []
  last_idx = -1
  for i in range(num_timestamps):
    lo = max(edges[i], last_idx + 1)
    hi = max(lo + 1, edges[i + 1])
    hi = min(hi, len(groups))
    idx = rng.randrange(lo, hi)
    selected.append(groups[idx])
    last_idx = idx
  return selected


def split_views(
    views: list[ViewRef],
    rng: random.Random,
) -> tuple[list[ViewRef], list[ViewRef]]:
  shuffled = list(views)
  rng.shuffle(shuffled)
  if len(shuffled) == 2:
    return [shuffled[0]], [shuffled[1]]
  pivot = max(1, len(shuffled) // 2)
  return shuffled[:pivot], shuffled[pivot:]


def build_role_batch(
    role_tracks: list[list[Group]],
    role: str,
    episodes_per_role: int,
    num_timestamps: int,
    rng: random.Random,
) -> list[tuple[str, str, int, list[ViewRef], list[ViewRef]]]:
  pairs = []
  for _ in range(episodes_per_role):
    groups = rng.choice(role_tracks)
    sampled = stratified_group_sample(groups, num_timestamps, rng)
    for group in sampled:
      subset_a, subset_b = split_views(group.views, rng)
      pairs.append((group.episode_id, role, group.timestamp_ms, subset_a, subset_b))
  return pairs


def load_image(root: Path, rel_path: str, transform) -> torch.Tensor:
  image = Image.open(root / rel_path).convert("RGB")
  return transform(image)


def prepare_images(
    root: Path,
    pairs: list[tuple[str, str, int, list[ViewRef], list[ViewRef]]],
    transform,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  images = []
  pair_indices = []
  subset_indices = []
  camera_ids = []
  for pair_idx, (_, _, _, subset_a, subset_b) in enumerate(pairs):
    for subset_idx, subset in enumerate([subset_a, subset_b]):
      for view in subset:
        images.append(load_image(root, view.rel_path, transform))
        pair_indices.append(pair_idx)
        subset_indices.append(subset_idx)
        camera_ids.append(view.camera_id)
  return (
      torch.stack(images, dim=0).to(device),
      torch.tensor(pair_indices, dtype=torch.long, device=device),
      torch.tensor(subset_indices, dtype=torch.long, device=device),
      torch.tensor(camera_ids, dtype=torch.long, device=device),
  )


def siglip_loss(z_a, z_b, logit_scale, logit_bias):
  logits = logit_scale.exp() * (z_a @ z_b.t()) + logit_bias
  labels = 2.0 * torch.eye(logits.shape[0], device=logits.device) - 1.0
  loss = -F.logsigmoid(labels * logits).mean()
  sims = z_a @ z_b.t()
  top1_ab = (sims.argmax(dim=1) == torch.arange(
      sims.shape[0], device=sims.device)).float().mean()
  top1_ba = (sims.argmax(dim=0) == torch.arange(
      sims.shape[0], device=sims.device)).float().mean()
  diag = sims.diag().mean()
  offdiag = (sims.sum() - sims.diag().sum()) / max(1, sims.numel() - sims.shape[0])
  return loss, {
      "top1_ab": top1_ab,
      "top1_ba": top1_ba,
      "diag": diag,
      "offdiag": offdiag,
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

  by_role = load_groups(args.index, args.num_timestamps, args.max_groups)
  print(
      f"loaded tracks h={len(by_role['h'])} r={len(by_role['r'])} "
      f"num_timestamps={args.num_timestamps}")
  if not by_role["h"] or not by_role["r"]:
    raise RuntimeError("Need both human and robot tracks.")

  transform = make_transform(args.image_size)
  model = FixedSlotMultiViewVVCL(
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
        "loss",
        "loss_h",
        "loss_r",
        "top1_h",
        "top1_r",
        "diag_h",
        "offdiag_h",
        "diag_r",
        "offdiag_r",
        "pairs_h",
        "pairs_r",
        "images",
        "logit_scale",
        "logit_bias",
        "seconds",
        "cuda_mem_mb",
    ])
    f.flush()

    start = time.time()
    for step in range(1, args.max_iters + 1):
      iter_start = time.time()
      human_pairs = build_role_batch(
          by_role["h"],
          "h",
          args.episodes_per_role,
          args.num_timestamps,
          rng,
      )
      robot_pairs = build_role_batch(
          by_role["r"],
          "r",
          args.episodes_per_role,
          args.num_timestamps,
          rng,
      )
      all_pairs = human_pairs + robot_pairs
      images, pair_idx, subset_idx, camera_ids = prepare_images(
          args.tcc_root, all_pairs, transform, device)

      optimizer.zero_grad(set_to_none=True)
      with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
        z = model.encode_subsets(
            images,
            pair_idx,
            subset_idx,
            camera_ids,
            num_pairs=len(all_pairs),
        )
        n_h = len(human_pairs)
        z_h = z[:n_h]
        z_r = z[n_h:]
        loss_h, metrics_h = siglip_loss(
            z_h[:, 0],
            z_h[:, 1],
            model.logit_scale,
            model.logit_bias,
        )
        loss_r, metrics_r = siglip_loss(
            z_r[:, 0],
            z_r[:, 1],
            model.logit_scale,
            model.logit_bias,
        )
        loss = 0.5 * (loss_h + loss_r)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      seconds = time.time() - iter_start
      mem_mb = (
          torch.cuda.max_memory_allocated(device) / 1024 / 1024
          if device.type == "cuda" else 0.0)
      row = [
          step,
          f"{loss.item():.8f}",
          f"{loss_h.item():.8f}",
          f"{loss_r.item():.8f}",
          f"{0.5 * (metrics_h['top1_ab'].item() + metrics_h['top1_ba'].item()):.6f}",
          f"{0.5 * (metrics_r['top1_ab'].item() + metrics_r['top1_ba'].item()):.6f}",
          f"{metrics_h['diag'].item():.6f}",
          f"{metrics_h['offdiag'].item():.6f}",
          f"{metrics_r['diag'].item():.6f}",
          f"{metrics_r['offdiag'].item():.6f}",
          n_h,
          len(robot_pairs),
          images.shape[0],
          f"{model.logit_scale.exp().item():.6f}",
          f"{model.logit_bias.item():.6f}",
          f"{seconds:.4f}",
          f"{mem_mb:.1f}",
      ]
      writer.writerow(row)
      if step == 1 or step % args.log_every == 0:
        print(
            f"step {step:05d} loss={row[1]} h={row[2]} r={row[3]} "
            f"top1_h={row[4]} top1_r={row[5]} "
            f"diag/off_h={row[6]}/{row[7]} diag/off_r={row[8]}/{row[9]} "
            f"pairs={n_h}+{len(robot_pairs)} imgs={images.shape[0]} "
            f"scale={row[13]} bias={row[14]} mem={row[16]}MB sec={row[15]}",
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
