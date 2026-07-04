#!/usr/bin/env python3
"""Train timestamp multi-view fusion with H/R Soft-DTW contrastive loss."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from xirl.losses import soft_dtw_sequence_distance
from xirl.models import build_backbone


PATH_ARG_NAMES = {
    "data_root",
    "timestamp_groups",
    "training_index",
    "out_dir",
    "output_root",
    "pretrain_path",
}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/default.yaml"


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
  """Frozen ViT backbone + fixed camera slot fusion with separate loss heads."""

  def __init__(
      self,
      num_camera_slots: int,
      embedding_size: int,
      fusion_size: int,
      pretrain_path: str,
      backbone: str = "vit_b16",
      train_layernorm: bool = True,
      train_adapters: bool = False,
  ):
    super().__init__()
    self.num_camera_slots = num_camera_slots
    self.backbone = build_backbone(
        backbone=backbone,
        pretrain_path=pretrain_path,
        train_norm_affine=train_layernorm,
        train_adapters=train_adapters,
    )

    input_dim = num_camera_slots * self.backbone.output_dim + num_camera_slots
    self.fusion = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, fusion_size),
        nn.GELU(),
        nn.Linear(fusion_size, fusion_size),
        nn.GELU(),
    )
    self.projector_sdtw = nn.Linear(fusion_size, embedding_size)
    self.projector_aux = nn.Linear(fusion_size, embedding_size)

  def encode_groups(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    feats = self.backbone(images)
    feats = torch.flatten(feats, 1)
    slots = feats.new_zeros(
        (num_groups, self.num_camera_slots, feats.shape[-1]))
    masks = feats.new_zeros((num_groups, self.num_camera_slots))
    slots[group_indices, camera_ids] = feats
    masks[group_indices, camera_ids] = 1.0
    fused_input = torch.cat([slots.flatten(start_dim=1), masks], dim=-1)
    fused = self.fusion(fused_input)
    z_sdtw = self.projector_sdtw(fused)
    z_aux = self.projector_aux(fused)
    return F.normalize(z_sdtw, dim=-1), F.normalize(z_aux, dim=-1)

  def encode_group_subsets(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      subset_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats = self.backbone(images)
    feats = torch.flatten(feats, 1)
    subset_slots = feats.new_zeros(
        (num_groups, 2, self.num_camera_slots, feats.shape[-1]))
    subset_masks = feats.new_zeros((num_groups, 2, self.num_camera_slots))
    subset_slots[group_indices, subset_indices, camera_ids] = feats
    subset_masks[group_indices, subset_indices, camera_ids] = 1.0
    subset_input = torch.cat([
        subset_slots.flatten(start_dim=2),
        subset_masks,
    ], dim=-1)
    subset_fused = self.fusion(subset_input)
    z_subset_sdtw = self.projector_sdtw(subset_fused)
    z_subset_aux = self.projector_aux(subset_fused)

    full_slots = feats.new_zeros(
        (num_groups, self.num_camera_slots, feats.shape[-1]))
    full_masks = feats.new_zeros((num_groups, self.num_camera_slots))
    full_slots[group_indices, camera_ids] = feats
    full_masks[group_indices, camera_ids] = 1.0
    full_input = torch.cat([full_slots.flatten(start_dim=1), full_masks], dim=-1)
    full_fused = self.fusion(full_input)
    z_full_sdtw = self.projector_sdtw(full_fused)
    return (
        F.normalize(z_full_sdtw, dim=-1),
        F.normalize(z_subset_sdtw, dim=-1),
        F.normalize(z_subset_aux, dim=-1),
    )

  def forward(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      subset_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self.encode_group_subsets(
        images, group_indices, subset_indices, camera_ids, num_groups)


class ViewSetAttentionSoftDTW(nn.Module):
  """Frozen ViT backbone + unordered view-set attention pooling."""

  def __init__(
      self,
      embedding_size: int,
      fusion_size: int,
      pretrain_path: str,
      backbone: str = "vit_b16",
      train_layernorm: bool = True,
      train_adapters: bool = False,
      view_token_dropout: float = 0.15,
      view_token_noise_std: float = 0.01,
      attention_dropout: float = 0.1,
      projector_dropout: float = 0.1,
  ):
    super().__init__()
    self.view_token_dropout = view_token_dropout
    self.view_token_noise_std = view_token_noise_std
    self.attention_dropout = attention_dropout
    self.backbone = build_backbone(
        backbone=backbone,
        pretrain_path=pretrain_path,
        train_norm_affine=train_layernorm,
        train_adapters=train_adapters,
    )

    self.view_projector = nn.Sequential(
        nn.LayerNorm(self.backbone.output_dim),
        nn.Linear(self.backbone.output_dim, fusion_size),
        nn.GELU(),
        nn.Linear(fusion_size, fusion_size),
        nn.GELU(),
    )
    self.view_norm = nn.LayerNorm(fusion_size)
    self.query = nn.Parameter(torch.randn(fusion_size) * 0.02)
    self.pooled_norm = nn.LayerNorm(fusion_size)
    self.projector_sdtw = nn.Linear(fusion_size, embedding_size)
    self.projector_aux = nn.Sequential(
        nn.Dropout(projector_dropout),
        nn.Linear(fusion_size, embedding_size),
    )

  def _pack_view_tokens(
      self,
      tokens_flat: torch.Tensor,
      group_indices: torch.Tensor,
      num_groups: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(group_indices, minlength=num_groups)
    max_views = int(counts.max().item())
    tokens = tokens_flat.new_zeros((num_groups, max_views, tokens_flat.shape[-1]))
    mask = torch.zeros(
        (num_groups, max_views),
        dtype=torch.bool,
        device=tokens_flat.device,
    )
    order = torch.argsort(group_indices)
    sorted_groups = group_indices[order]
    sorted_tokens = tokens_flat[order]
    group_starts = torch.cumsum(counts, dim=0) - counts
    positions = (
        torch.arange(sorted_groups.shape[0], device=tokens_flat.device)
        - group_starts[sorted_groups]
    )
    tokens[sorted_groups, positions] = sorted_tokens
    mask[sorted_groups, positions] = True
    return tokens, mask

  def encode_view_tokens(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      num_groups: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    feats = self.backbone(images)
    feats = torch.flatten(feats, 1)
    tokens_flat = self.view_norm(self.view_projector(feats))
    return self._pack_view_tokens(tokens_flat, group_indices, num_groups)

  def augment_view_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
    out = tokens
    if self.training and self.view_token_dropout > 0:
      out = F.dropout(out, p=self.view_token_dropout, training=True)
    if self.training and self.view_token_noise_std > 0:
      out = out + torch.randn_like(out) * self.view_token_noise_std
    return out

  def attention_pool(
      self,
      tokens: torch.Tensor,
      mask: torch.Tensor,
      apply_dropout: bool,
  ) -> torch.Tensor:
    scale = tokens.shape[-1] ** -0.5
    scores = (tokens * self.query).sum(dim=-1) * scale
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    weights = weights.masked_fill(~mask, 0.0)
    if self.training and apply_dropout and self.attention_dropout > 0:
      dropped = F.dropout(weights, p=self.attention_dropout, training=True)
      denom = dropped.sum(dim=-1, keepdim=True)
      weights = torch.where(denom > 0, dropped / denom.clamp_min(1e-6), weights)
    pooled = (weights.unsqueeze(-1) * tokens).sum(dim=1)
    return self.pooled_norm(pooled)

  def sample_feature_view_mask(
      self,
      mask: torch.Tensor,
      mode: str,
      mask_prob: float,
      min_views: int,
  ) -> torch.Tensor:
    if (not self.training) or mode == "none" or mask_prob <= 0:
      return mask
    if mode != "drop_one":
      raise ValueError(f"Unknown feature view mask mode: {mode}")
    del min_views
    out = mask.clone()
    for group_idx in range(out.shape[0]):
      valid = torch.nonzero(out[group_idx], as_tuple=False).flatten()
      if valid.numel() <= 1:
        continue
      if torch.rand((), device=out.device) >= mask_prob:
        continue
      drop_pos = torch.randint(valid.numel(), (1,), device=out.device)
      out[group_idx, valid[drop_pos]] = False
    return out

  def encode_groups(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del camera_ids
    tokens, mask = self.encode_view_tokens(images, group_indices, num_groups)
    pooled = self.attention_pool(tokens, mask, apply_dropout=False)
    z_sdtw = self.projector_sdtw(pooled)
    z_aux = self.projector_aux(pooled)
    return F.normalize(z_sdtw, dim=-1), F.normalize(z_aux, dim=-1)

  def forward(
      self,
      images: torch.Tensor,
      group_indices: torch.Tensor,
      camera_ids: torch.Tensor,
      num_groups: int,
      aug_images: torch.Tensor | None = None,
      aug_group_indices: torch.Tensor | None = None,
      aug_camera_ids: torch.Tensor | None = None,
      aug_num_groups: int | None = None,
      feature_view_mask_mode: str = "none",
      feature_view_mask_prob: float = 0.0,
      feature_view_mask_min_views: int = 1,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del camera_ids
    tokens, mask = self.encode_view_tokens(images, group_indices, num_groups)
    clean = self.attention_pool(tokens, mask, apply_dropout=False)
    z_sdtw = self.projector_sdtw(clean)
    z_aux_clean = self.projector_aux(clean)
    if aug_images is None:
      if feature_view_mask_mode == "none":
        aug_a = self.attention_pool(
            self.augment_view_tokens(tokens), mask, apply_dropout=True)
        aug_b = self.attention_pool(
            self.augment_view_tokens(tokens), mask, apply_dropout=True)
        z_aux_clean = self.projector_aux(aug_a)
        z_aux_aug = self.projector_aux(aug_b)
      else:
        aug_mask = self.sample_feature_view_mask(
            mask,
            feature_view_mask_mode,
            feature_view_mask_prob,
            feature_view_mask_min_views,
        )
        aug = self.attention_pool(
            self.augment_view_tokens(tokens), aug_mask, apply_dropout=True)
        z_aux_aug = self.projector_aux(aug)
    else:
      if aug_group_indices is None or aug_num_groups is None:
        raise ValueError("Augmented view-set inputs require group indices.")
      del aug_camera_ids
      aug_tokens, aug_mask = self.encode_view_tokens(
          aug_images, aug_group_indices, aug_num_groups)
      aug = self.attention_pool(aug_tokens, aug_mask, apply_dropout=True)
      z_aux_aug = self.projector_aux(aug)
    return (
        F.normalize(z_sdtw, dim=-1),
        F.normalize(z_aux_clean, dim=-1),
        F.normalize(z_aux_aug, dim=-1),
    )


def load_config(path: Path) -> dict:
  with path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
  if config is None:
    return {}
  if not isinstance(config, dict):
    raise ValueError(f"Config must be a YAML mapping: {path}")
  return config


def normalize_config_keys(config: dict) -> dict:
  normalized = {key.replace("-", "_"): value for key, value in config.items()}
  if "batch_pairs" in normalized and "batch_episode_pairs" not in normalized:
    normalized["batch_episode_pairs"] = normalized.pop("batch_pairs")
  return normalized


def coerce_path_args(args: argparse.Namespace) -> argparse.Namespace:
  for name in PATH_ARG_NAMES:
    if not hasattr(args, name):
      continue
    value = getattr(args, name)
    if value is not None and not isinstance(value, Path):
      setattr(args, name, Path(value))
  return args


def build_parser(parents=None) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      parents=parents or [],
      argument_default=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--data-root",
      "--tcc-root",
      dest="data_root",
      type=Path,
  )
  parser.add_argument(
      "--timestamp-groups",
      "--index",
      dest="timestamp_groups",
      type=Path,
  )
  parser.add_argument(
      "--training-index",
      "--matched-index",
      dest="training_index",
      type=Path,
      help="Universal matched-camera training index.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
  )
  parser.add_argument("--output-root", type=Path)
  parser.add_argument("--run-name")
  parser.add_argument("--num-timestamps", type=int)
  parser.add_argument(
      "--num-multi-view",
      dest="num_multi_view",
      type=int,
      help="Number of timestamp-aligned camera views used per group.",
  )
  parser.add_argument(
      "--batch-episode-pairs",
      "--batch-pairs",
      dest="batch_episode_pairs",
      type=int,
      help="Number of episode-level H/R pairs sampled per GPU/process.",
  )
  parser.add_argument("--max-iters", type=int)
  parser.add_argument("--log-every", type=int)
  parser.add_argument("--save-every", type=int)
  parser.add_argument("--gamma", type=float)
  parser.add_argument("--temperature", type=float)
  parser.add_argument(
      "--softdtw-divergence",
      action=argparse.BooleanOptionalAction,
      dest="softdtw_divergence",
      default=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--softdtw-mode",
      choices=("contrastive", "paired", "tcc", "soft_alignment"),
      help=(
          "contrastive computes the full H/R batch distance matrix; paired "
          "only aligns each episode-level H/R pair; tcc uses paired cycle-back; "
          "soft_alignment distills a Sinkhorn temporal-prior teacher."
      ),
  )
  parser.add_argument("--soft-alignment-temperature", type=float)
  parser.add_argument("--soft-alignment-epsilon", type=float)
  parser.add_argument("--soft-alignment-rho", type=float)
  parser.add_argument("--soft-alignment-sinkhorn-iters", type=int)
  parser.add_argument("--soft-alignment-struct-lambda", type=float)
  parser.add_argument("--soft-alignment-max-forward-step", type=float)
  parser.add_argument("--lambda-mv", type=float)
  parser.add_argument("--mv-temperature", type=float)
  parser.add_argument(
      "--mv-soft-temporal",
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help=(
          "Use timestamp-index soft targets for auxiliary multi-view InfoNCE."
      ),
  )
  parser.add_argument("--mv-soft-temporal-alpha", type=float)
  parser.add_argument("--mv-soft-temporal-tau", type=float)
  parser.add_argument(
      "--tcc-loss-type",
      choices=(
          "classification",
          "regression_mse",
          "regression_mse_var",
          "regression_huber",
      ),
  )
  parser.add_argument(
      "--tcc-similarity-type",
      choices=("l2", "cosine"),
  )
  parser.add_argument("--tcc-label-smoothing", type=float)
  parser.add_argument("--tcc-variance-lambda", type=float)
  parser.add_argument("--tcc-huber-delta", type=float)
  parser.add_argument(
      "--camera-matched-pairing",
      action=argparse.BooleanOptionalAction,
      dest="camera_matched_pairing",
      default=argparse.SUPPRESS,
      help=(
          "Require H/R to sample the same valid camera set inside each "
          "paired episode. The camera set size is --num-multi-view."
      ),
  )
  parser.add_argument("--lr", type=float)
  parser.add_argument("--weight-decay", type=float)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--image-size", type=int)
  parser.add_argument("--device")
  parser.add_argument("--embedding-size", type=int)
  parser.add_argument("--fusion-size", type=int)
  parser.add_argument(
      "--fusion-mode",
      choices=("fixed_slot", "attention_pool"),
      help=(
          "Fusion module. fixed_slot uses camera-id slots; attention_pool "
          "treats sampled views as an unordered set and ignores camera ids."
      ),
  )
  parser.add_argument("--num-camera-slots", type=int)
  parser.add_argument("--view-token-dropout", type=float)
  parser.add_argument("--view-token-noise-std", type=float)
  parser.add_argument("--attention-dropout", type=float)
  parser.add_argument("--projector-dropout", type=float)
  parser.add_argument(
      "--pixel-aug",
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help="Use weak pixel-level augmentation for the auxiliary fused CL branch.",
  )
  parser.add_argument(
      "--view-mask-mode",
      choices=("none", "drop_one"),
      help="Stochastic view masking for the augmented fused CL branch.",
  )
  parser.add_argument("--view-mask-prob", type=float)
  parser.add_argument("--view-mask-min-views", type=int)
  parser.add_argument(
      "--backbone",
      choices=(
          "vit_b16",
          "vit",
          "r3m_resnet50",
          "r3m",
          "unadapted_r3m",
          "r3m_late_adapter",
          "r3m_adapter",
          "r3m_hralign_style",
          "r3m_align_l",
          "hralign_r3m_l",
          "adapted_r3m",
      ),
      help=(
          "Visual backbone. r3m_resnet50 uses original R3M without adapters; "
          "r3m_late_adapter inserts trainable HR-Align-style late adapters "
          "on original R3M; r3m_align_l loads AdaptedR3M.pyth."
      ),
  )
  parser.add_argument(
      "--train-backbone-adapters",
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
      help="Also fine-tune HR-Align late adapter weights for R3M backbones.",
  )
  parser.add_argument("--pretrain-path")
  parser.add_argument(
      "--amp",
      action=argparse.BooleanOptionalAction,
      default=argparse.SUPPRESS,
  )
  return parser


def parse_args() -> argparse.Namespace:
  config_parser = argparse.ArgumentParser(add_help=False)
  config_parser.add_argument(
      "--config",
      type=Path,
      help="YAML config file. CLI arguments override config values.",
  )
  config_args, _ = config_parser.parse_known_args()
  parser = build_parser(parents=[config_parser])
  valid_keys = {
      action.dest for action in parser._actions  # pylint: disable=protected-access
  }

  merged_config = {}
  config_sources = []
  if DEFAULT_CONFIG_PATH.exists():
    config_sources.append(DEFAULT_CONFIG_PATH)
  if config_args.config is not None:
    config_sources.append(config_args.config)

  for config_path in config_sources:
    config = normalize_config_keys(load_config(config_path))
    unknown = sorted(set(config) - valid_keys)
    if unknown:
      raise ValueError(
          f"Unknown config keys in {config_path}: {unknown}")
    merged_config.update(config)

  cli_args = vars(parser.parse_args())
  merged_config.update(cli_args)
  if "config" not in merged_config:
    merged_config["config"] = config_args.config

  args = coerce_path_args(argparse.Namespace(**merged_config))
  if not hasattr(args, "out_dir"):
    if not hasattr(args, "output_root") or not hasattr(args, "run_name"):
      raise ValueError(
          "Config must provide either out_dir or both output_root and run_name.")
    args.out_dir = args.output_root / args.run_name
  if not hasattr(args, "fusion_mode"):
    args.fusion_mode = "fixed_slot"
  if not hasattr(args, "softdtw_mode"):
    args.softdtw_mode = "contrastive"
  if not hasattr(args, "mv_soft_temporal"):
    args.mv_soft_temporal = False
  if not hasattr(args, "mv_soft_temporal_alpha"):
    args.mv_soft_temporal_alpha = 0.2
  if not hasattr(args, "mv_soft_temporal_tau"):
    args.mv_soft_temporal_tau = 1.0
  if not hasattr(args, "tcc_loss_type"):
    args.tcc_loss_type = "regression_mse"
  if not hasattr(args, "tcc_similarity_type"):
    args.tcc_similarity_type = "l2"
  if not hasattr(args, "tcc_label_smoothing"):
    args.tcc_label_smoothing = 0.1
  if not hasattr(args, "tcc_variance_lambda"):
    args.tcc_variance_lambda = 0.001
  if not hasattr(args, "tcc_huber_delta"):
    args.tcc_huber_delta = 0.1
  if not hasattr(args, "soft_alignment_temperature"):
    args.soft_alignment_temperature = getattr(args, "temperature", 0.1)
  if not hasattr(args, "soft_alignment_epsilon"):
    args.soft_alignment_epsilon = 0.05
  if not hasattr(args, "soft_alignment_rho"):
    args.soft_alignment_rho = 0.5
  if not hasattr(args, "soft_alignment_sinkhorn_iters"):
    args.soft_alignment_sinkhorn_iters = 20
  if not hasattr(args, "soft_alignment_struct_lambda"):
    args.soft_alignment_struct_lambda = 0.1
  if not hasattr(args, "soft_alignment_max_forward_step"):
    args.soft_alignment_max_forward_step = 1.0
  if not hasattr(args, "view_token_dropout"):
    args.view_token_dropout = 0.15
  if not hasattr(args, "view_token_noise_std"):
    args.view_token_noise_std = 0.01
  if not hasattr(args, "attention_dropout"):
    args.attention_dropout = 0.1
  if not hasattr(args, "projector_dropout"):
    args.projector_dropout = 0.1
  if not hasattr(args, "pixel_aug"):
    args.pixel_aug = False
  if not hasattr(args, "view_mask_mode"):
    args.view_mask_mode = "none"
  if not hasattr(args, "view_mask_prob"):
    args.view_mask_prob = 0.0
  if not hasattr(args, "view_mask_min_views"):
    args.view_mask_min_views = 1
  if not hasattr(args, "backbone"):
    args.backbone = "vit_b16"
  if not hasattr(args, "train_backbone_adapters"):
    args.train_backbone_adapters = False
  return coerce_path_args(args)


def normalize_max_groups(max_groups: int | None) -> int | None:
  if max_groups is None or max_groups <= 0:
    return None
  return max_groups


def load_tracks(
    index_path: Path,
    min_groups: int,
    num_multi_view: int,
    max_groups: int | None,
    camera_matched_pairing: bool,
    combo_size: int,
) -> dict[str, dict[str, list[Group]]]:
  tracks: dict[tuple[str, str], list[Group]] = defaultdict(list)
  loaded = 0
  with index_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      views_raw = json.loads(row["views_json"])
      if len(views_raw) < num_multi_view:
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
    if len(h) < min_groups or len(r) < min_groups:
      continue
    if camera_matched_pairing:
      shared_combos = find_shared_valid_camera_combos(
          h, r, combo_size=combo_size, min_groups=min_groups)
      if not shared_combos:
        continue
      paired[episode_id] = {
          "h": h,
          "r": r,
          "shared_combos": shared_combos,
      }
    else:
      paired[episode_id] = {"h": h, "r": r}
  return paired


def load_matched_training_index(
    index_path: Path,
    min_groups: int,
    combo_size: int,
) -> dict[str, dict[str, list[Group]]]:
  payload = torch.load(index_path, map_location="cpu", weights_only=False)
  if payload.get("format") not in {
      "matched_combo_training_index_v1",
      "matched_combo_training_index_v2",
      "matched_combo_universal_group_pool_v1",
  }:
    raise ValueError(f"Unsupported matched index format: {payload.get('format')}")
  if payload["format"] != "matched_combo_universal_group_pool_v1" and int(payload["num_timestamps"]) < min_groups:
    raise ValueError(
        f"Matched index only supports {payload['num_timestamps']} timestamps, "
        f"but {min_groups} were requested.")
  if payload["format"] != "matched_combo_universal_group_pool_v1" and int(payload["num_views"]) != combo_size:
    raise ValueError(
        f"Matched index has {payload['num_views']} views, "
        f"but --num-multi-view={combo_size}.")

  paired = {}
  for episode_id, rec in payload["episodes"].items():
    if payload["format"] == "matched_combo_universal_group_pool_v1":
      shared_combos = find_shared_valid_pool_combos(
          rec["h_group_pool"],
          rec["r_group_pool"],
          combo_size=combo_size,
          min_groups=min_groups,
      )
      if shared_combos:
        paired[episode_id] = {
            "task_id": rec["task_id"],
            "h_group_pool": rec["h_group_pool"],
            "r_group_pool": rec["r_group_pool"],
            "shared_combos": shared_combos,
            "lazy_matched_index": True,
        }
      continue

    if payload["format"] == "matched_combo_training_index_v2":
      shared_combos = [
          {
              "camera_ids": tuple(
                  int(camera_id) for camera_id in combo_rec["camera_ids"]),
              "h_group_indices": combo_rec["h_group_indices"],
              "r_group_indices": combo_rec["r_group_indices"],
          }
          for combo_rec in rec["shared_combos"]
          if (
              len(combo_rec["h_group_indices"]) >= min_groups
              and len(combo_rec["r_group_indices"]) >= min_groups
          )
      ]
      if shared_combos:
        paired[episode_id] = {
            "task_id": rec["task_id"],
            "h_group_pool": rec["h_group_pool"],
            "r_group_pool": rec["r_group_pool"],
            "shared_combos": shared_combos,
            "lazy_matched_index": True,
        }
      continue

    shared_combos = []
    for combo_rec in rec["shared_combos"]:
      combo = tuple(int(camera_id) for camera_id in combo_rec["camera_ids"])
      h_groups = [
          Group(
              episode_id=episode_id,
              task_id=rec["task_id"],
              role="h",
              timestamp_ms=int(group["timestamp_ms"]),
              views=[
                  ViewRef(
                      rel_path=view["rel_path"],
                      camera_id=int(view["camera_id"]),
                  )
                  for view in group["views"]
              ],
          )
          for group in combo_rec["h_groups"]
      ]
      r_groups = [
          Group(
              episode_id=episode_id,
              task_id=rec["task_id"],
              role="r",
              timestamp_ms=int(group["timestamp_ms"]),
              views=[
                  ViewRef(
                      rel_path=view["rel_path"],
                      camera_id=int(view["camera_id"]),
                  )
                  for view in group["views"]
              ],
          )
          for group in combo_rec["r_groups"]
      ]
      if len(h_groups) >= min_groups and len(r_groups) >= min_groups:
        shared_combos.append({
            "camera_ids": combo,
            "h": h_groups,
            "r": r_groups,
        })
    if shared_combos:
      paired[episode_id] = {
          "h": shared_combos[0]["h"],
          "r": shared_combos[0]["r"],
          "shared_combos": shared_combos,
      }
  return paired


def materialize_combo_groups(
    group_pool: list[dict],
    group_indices: list[int],
    combo: tuple[int, ...],
    episode_id: str,
    task_id: str,
    role: str,
) -> list[Group]:
  groups = []
  for group_idx in group_indices:
    group = group_pool[int(group_idx)]
    views_by_camera = {
        int(camera_id): view
        for camera_id, view in group["views_by_camera"].items()
    }
    groups.append(Group(
        episode_id=episode_id,
        task_id=task_id,
        role=role,
        timestamp_ms=int(group["timestamp_ms"]),
        views=[
            ViewRef(
                rel_path=views_by_camera[camera_id]["rel_path"],
                camera_id=int(views_by_camera[camera_id]["camera_id"]),
            )
            for camera_id in combo
        ],
    ))
  return groups


def build_pool_combo_indices(
    group_pool: list[dict],
    combo_size: int,
) -> dict[tuple[int, ...], list[int]]:
  combo_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
  for idx, group in enumerate(group_pool):
    camera_ids = sorted(int(camera_id) for camera_id in group["views_by_camera"])
    if len(camera_ids) < combo_size:
      continue
    for combo in itertools.combinations(camera_ids, combo_size):
      combo_indices[combo].append(idx)
  return combo_indices


def find_shared_valid_pool_combos(
    h_pool: list[dict],
    r_pool: list[dict],
    combo_size: int,
    min_groups: int,
) -> list[dict]:
  h_combo_indices = build_pool_combo_indices(h_pool, combo_size)
  r_combo_indices = build_pool_combo_indices(r_pool, combo_size)
  h_valid = {
      combo for combo, indices in h_combo_indices.items()
      if len(indices) >= min_groups
  }
  r_valid = {
      combo for combo, indices in r_combo_indices.items()
      if len(indices) >= min_groups
  }
  shared = sorted(h_valid & r_valid)
  out = []
  for combo in shared:
    h_indices = h_combo_indices[combo]
    r_indices = r_combo_indices[combo]
    if len(h_indices) >= min_groups and len(r_indices) >= min_groups:
      out.append({
          "camera_ids": combo,
          "h_group_indices": h_indices,
          "r_group_indices": r_indices,
      })
  return out


def count_camera_combos(
    groups: list[Group],
    combo_size: int,
) -> Counter[tuple[int, ...]]:
  counts: Counter[tuple[int, ...]] = Counter()
  for group in groups:
    camera_ids = sorted({view.camera_id for view in group.views})
    if len(camera_ids) < combo_size:
      continue
    for combo in itertools.combinations(camera_ids, combo_size):
      counts[combo] += 1
  return counts


def find_shared_valid_camera_combos(
    human_groups: list[Group],
    robot_groups: list[Group],
    combo_size: int,
    min_groups: int,
) -> list[tuple[int, ...]]:
  h_counts = count_camera_combos(human_groups, combo_size)
  r_counts = count_camera_combos(robot_groups, combo_size)
  h_valid = {combo for combo, count in h_counts.items() if count >= min_groups}
  r_valid = {combo for combo, count in r_counts.items() if count >= min_groups}
  return sorted(h_valid & r_valid)


def filter_groups_to_camera_combo(
    groups: list[Group],
    combo: tuple[int, ...],
) -> list[Group]:
  out = []
  for group in groups:
    by_camera = {view.camera_id: view for view in group.views}
    if not all(camera_id in by_camera for camera_id in combo):
      continue
    out.append(Group(
        episode_id=group.episode_id,
        task_id=group.task_id,
        role=group.role,
        timestamp_ms=group.timestamp_ms,
        views=[by_camera[camera_id] for camera_id in combo],
    ))
  return out


def build_task_to_episodes(
    paired_tracks: dict[str, dict[str, list[Group]]],
) -> dict[str, list[str]]:
  task_to_episodes: dict[str, list[str]] = defaultdict(list)
  for episode_id, tracks in paired_tracks.items():
    task_id = tracks.get("task_id")
    if task_id is None:
      task_id = tracks["h"][0].task_id
    task_to_episodes[task_id].append(episode_id)
  return task_to_episodes


def collect_group_camera_ids(group) -> set[int]:
  if isinstance(group, Group):
    return {view.camera_id for view in group.views}
  if isinstance(group, dict):
    if "views_by_camera" in group:
      return {int(camera_id) for camera_id in group["views_by_camera"]}
    if "views" in group:
      return {int(view["camera_id"]) for view in group["views"]}
  return set()


def infer_num_camera_slots(paired_tracks: dict[str, dict]) -> int:
  camera_ids: set[int] = set()
  for tracks in paired_tracks.values():
    for key in ["h", "r", "h_group_pool", "r_group_pool"]:
      for group in tracks.get(key, []):
        camera_ids.update(collect_group_camera_ids(group))
    for combo_rec in tracks.get("shared_combos", []):
      if isinstance(combo_rec, dict):
        if "camera_ids" in combo_rec:
          camera_ids.update(int(camera_id) for camera_id in combo_rec["camera_ids"])
        for key in ["h", "r"]:
          for group in combo_rec.get(key, []):
            camera_ids.update(collect_group_camera_ids(group))
      else:
        camera_ids.update(int(camera_id) for camera_id in combo_rec)
  if not camera_ids:
    raise ValueError("Could not infer camera slots from paired tracks.")
  return max(camera_ids) + 1


def sample_episode_batch(
    episode_ids: list[str],
    task_to_episodes: dict[str, list[str]],
    batch_episode_pairs: int,
    prefer_distinct_tasks: bool,
    rng: random.Random,
) -> list[str]:
  if not prefer_distinct_tasks:
    return rng.sample(episode_ids, batch_episode_pairs)
  task_ids = [task_id for task_id, eps in task_to_episodes.items() if eps]
  selected_tasks = rng.sample(
      task_ids, min(batch_episode_pairs, len(task_ids)))
  selected = [rng.choice(task_to_episodes[task_id]) for task_id in selected_tasks]
  if len(selected) >= batch_episode_pairs:
    return selected

  selected_set = set(selected)
  remaining = [episode_id for episode_id in episode_ids if episode_id not in selected_set]
  rng.shuffle(remaining)
  selected.extend(remaining[:batch_episode_pairs - len(selected)])
  if len(selected) < batch_episode_pairs:
    selected.extend(
        rng.choices(episode_ids, k=batch_episode_pairs - len(selected)))
  return selected


def make_transform(image_size: int):
  return transforms.Compose([
      transforms.Resize((image_size, image_size), antialias=True),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225),
      ),
  ])


def make_weak_aug_transform(image_size: int):
  return transforms.Compose([
      transforms.Resize((image_size, image_size), antialias=True),
      transforms.ColorJitter(
          brightness=0.2,
          contrast=0.2,
          saturation=0.2,
          hue=0.05,
      ),
      transforms.RandomGrayscale(p=0.1),
      transforms.RandomApply([
          transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
      ], p=0.1),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225),
      ),
      transforms.RandomErasing(p=0.05, scale=(0.02, 0.08), value=0.0),
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
  with Image.open(root / rel_path) as image:
    return transform(image.convert("RGB"))


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
    disjoint: bool,
    rng: random.Random,
) -> list[list[tuple[list[ViewRef], list[ViewRef]]]]:
  out = []
  for sequence in sequences:
    subset_sequence = []
    for group in sequence:
      if disjoint:
        views = list(group.views)
        rng.shuffle(views)
        if max_views > 0:
          views = views[:max_views]
        if len(views) == 1:
          subset_a = views
          subset_b = views
        else:
          split = max(1, len(views) // 2)
          subset_a = views[:split]
          subset_b = views[split:]
          if not subset_b:
            subset_b = views[:1]
      else:
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


def sample_masked_views(
    views: list[ViewRef],
    mode: str,
    mask_prob: float,
    min_views: int,
    rng: random.Random,
) -> list[ViewRef]:
  del min_views
  if mode == "none" or rng.random() >= mask_prob:
    return list(views)
  if mode != "drop_one":
    raise ValueError(f"Unknown view_mask_mode: {mode}")
  if len(views) <= 1:
    return list(views)
  selected = list(views)
  drop_idx = rng.randrange(len(selected))
  del selected[drop_idx]
  return selected


def prepare_clean_aug_batch_images(
    root: Path,
    sequences: list[list[Group]],
    clean_transform,
    aug_transform,
    device,
    view_mask_mode: str,
    view_mask_prob: float,
    view_mask_min_views: int,
    rng: random.Random,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
  clean_images = []
  clean_group_indices = []
  clean_camera_ids = []
  aug_images = []
  aug_group_indices = []
  aug_camera_ids = []
  group_idx = 0
  for sequence in sequences:
    for group in sequence:
      for view in group.views:
        clean_images.append(load_image(root, view.rel_path, clean_transform))
        clean_group_indices.append(group_idx)
        clean_camera_ids.append(view.camera_id)
      masked_views = sample_masked_views(
          group.views,
          view_mask_mode,
          view_mask_prob,
          view_mask_min_views,
          rng,
      )
      for view in masked_views:
        aug_images.append(load_image(root, view.rel_path, aug_transform))
        aug_group_indices.append(group_idx)
        aug_camera_ids.append(view.camera_id)
      group_idx += 1
  return (
      torch.stack(clean_images, dim=0).to(device),
      torch.tensor(clean_group_indices, dtype=torch.long, device=device),
      torch.tensor(clean_camera_ids, dtype=torch.long, device=device),
      torch.stack(aug_images, dim=0).to(device),
      torch.tensor(aug_group_indices, dtype=torch.long, device=device),
      torch.tensor(aug_camera_ids, dtype=torch.long, device=device),
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
    softdtw_divergence,
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
      divergence=softdtw_divergence,
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
    softdtw_divergence,
):
  distances = soft_dtw_sequence_distance(
      h_seq,
      r_seq,
      gamma=gamma,
      normalize_dimension=False,
      divergence=softdtw_divergence,
      normalize_time=True,
  )
  return distances.mean(), {
      "distances": distances,
      "top1": torch.ones((), device=h_seq.device),
      "pos_dist": distances.mean(),
      "off_dist": torch.zeros((), device=h_seq.device),
  }


def sinkhorn_rows_cols(kernel, iters, eps=1e-8):
  transport = kernel
  for _ in range(iters):
    transport = transport / transport.sum(dim=2, keepdim=True).clamp_min(eps)
    transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(eps)
  return transport / transport.sum(dim=2, keepdim=True).clamp_min(eps)


def make_temporal_prior_cost(num_timestamps, device, dtype):
  indices = torch.arange(num_timestamps, device=device, dtype=dtype)
  distance = (indices[:, None] - indices[None]).abs()
  return distance / max(1, num_timestamps - 1)


def make_progress_transition_cost(
    num_timestamps,
    max_forward_step,
    device,
    dtype,
):
  indices = torch.arange(num_timestamps, device=device, dtype=dtype)
  delta = indices[None] - indices[:, None]
  max_forward = max_forward_step / max(1, num_timestamps - 1)
  delta = delta / max(1, num_timestamps - 1)
  backward = F.relu(-delta)
  large_forward = F.relu(delta - max_forward)
  return backward.pow(2) + large_forward.pow(2)


def compute_structural_progress_loss(prob, transition_cost):
  if prob.shape[1] <= 1:
    return prob.new_zeros(())
  prev_prob = prob[:, :-1]
  next_prob = prob[:, 1:]
  expected_cost = torch.einsum(
      "bik,kl,bil->bi", prev_prob, transition_cost, next_prob)
  return expected_cost.mean()


def compute_soft_alignment_direction(
    source_seq,
    target_seq,
    temperature,
    epsilon,
    rho,
    sinkhorn_iters,
    struct_lambda,
    max_forward_step,
):
  source = F.normalize(source_seq.float(), dim=-1)
  target = F.normalize(target_seq.float(), dim=-1)
  similarity = torch.einsum("btd,bsd->bts", source, target)
  pred_log_prob = F.log_softmax(similarity / temperature, dim=2)
  pred_prob = pred_log_prob.exp()

  cost = 1.0 - similarity
  temporal_prior = make_temporal_prior_cost(
      source.shape[1], source.device, source.dtype)
  teacher_cost = cost + rho * temporal_prior[None]
  with torch.no_grad():
    kernel = torch.exp(-teacher_cost / epsilon).clamp_min(1e-8)
    teacher = sinkhorn_rows_cols(kernel, sinkhorn_iters)

  loss_align = -(teacher * pred_log_prob).sum(dim=2).mean()
  transition_cost = make_progress_transition_cost(
      source.shape[1],
      max_forward_step,
      source.device,
      source.dtype,
  )
  loss_struct = compute_structural_progress_loss(pred_prob, transition_cost)
  loss = loss_align + struct_lambda * loss_struct

  labels = torch.arange(source.shape[1], device=source.device)
  top1 = (pred_prob.argmax(dim=2) == labels[None]).float().mean()
  diagonal = cost.diagonal(dim1=1, dim2=2)
  off = (cost.sum() - diagonal.sum()) / max(
      1, cost.numel() - source.shape[0] * source.shape[1])
  return loss, {
      "top1": top1,
      "pos_dist": diagonal.mean(),
      "off_dist": off,
      "loss_align": loss_align,
      "loss_struct": loss_struct,
  }


def compute_soft_alignment_paired(
    h_seq,
    r_seq,
    temperature,
    epsilon,
    rho,
    sinkhorn_iters,
    struct_lambda,
    max_forward_step,
):
  if h_seq.ndim != 3 or r_seq.ndim != 3:
    raise ValueError("Soft alignment sequences must have shape [B, T, D].")
  if h_seq.shape != r_seq.shape:
    raise ValueError("Soft alignment H/R sequences must have the same shape.")

  loss_hr, metrics_hr = compute_soft_alignment_direction(
      h_seq,
      r_seq,
      temperature,
      epsilon,
      rho,
      sinkhorn_iters,
      struct_lambda,
      max_forward_step,
  )
  loss_rh, metrics_rh = compute_soft_alignment_direction(
      r_seq,
      h_seq,
      temperature,
      epsilon,
      rho,
      sinkhorn_iters,
      struct_lambda,
      max_forward_step,
  )
  loss = 0.5 * (loss_hr + loss_rh)
  return loss, {
      "distances": None,
      "top1": 0.5 * (metrics_hr["top1"] + metrics_rh["top1"]),
      "pos_dist": 0.5 * (metrics_hr["pos_dist"] + metrics_rh["pos_dist"]),
      "off_dist": 0.5 * (metrics_hr["off_dist"] + metrics_rh["off_dist"]),
      "align_loss": 0.5 * (
          metrics_hr["loss_align"] + metrics_rh["loss_align"]),
      "struct_loss": 0.5 * (
          metrics_hr["loss_struct"] + metrics_rh["loss_struct"]),
  }


def tcc_scaled_similarity(
    emb1,
    emb2,
    similarity_type,
    temperature,
    normalize_dimension,
):
  if similarity_type == "l2":
    similarity = -torch.cdist(emb1, emb2).pow(2)
    if normalize_dimension:
      similarity = similarity / emb1.shape[-1]
  else:
    similarity = emb1 @ emb2.t()
  return similarity / temperature


def tcc_align_sequence_pair(
    emb1,
    emb2,
    similarity_type,
    temperature,
    normalize_dimension,
):
  sim_12 = tcc_scaled_similarity(
      emb1,
      emb2,
      similarity_type,
      temperature,
      normalize_dimension,
  )
  nn_embs = F.softmax(sim_12, dim=1) @ emb2
  logits = tcc_scaled_similarity(
      nn_embs,
      emb1,
      similarity_type,
      temperature,
      normalize_dimension,
  )
  labels = torch.arange(emb1.shape[0], device=emb1.device)
  return logits, labels


def tcc_regression_loss(
    logits,
    labels,
    loss_type,
    variance_lambda,
    huber_delta,
):
  num_timestamps = logits.shape[1]
  steps = torch.arange(num_timestamps, device=logits.device).float()
  steps = steps / float(num_timestamps)
  target = steps[labels]
  beta = F.softmax(logits, dim=1)
  pred = (steps[None] * beta).sum(dim=1)
  if loss_type == "regression_mse":
    return F.mse_loss(pred, target)
  if loss_type == "regression_huber":
    return F.huber_loss(pred, target, delta=huber_delta)
  pred_var = torch.log(((steps[None] - pred[:, None]).pow(2) * beta).sum(dim=1))
  err_sq = (target - pred).pow(2)
  return (torch.exp(-pred_var) * err_sq + variance_lambda * pred_var).mean()


def compute_tcc_cycleback(
    h_seq,
    r_seq,
    temperature,
    loss_type,
    similarity_type,
    label_smoothing,
    variance_lambda,
    huber_delta,
):
  if h_seq.ndim != 3 or r_seq.ndim != 3:
    raise ValueError("TCC sequences must have shape [B, T, D].")
  if h_seq.shape != r_seq.shape:
    raise ValueError("TCC H/R sequences must have the same shape.")

  batch_size, num_timestamps, _ = h_seq.shape
  logits_list = []
  labels_list = []
  for idx in range(batch_size):
    logits_hr, labels_hr = tcc_align_sequence_pair(
        h_seq[idx],
        r_seq[idx],
        similarity_type,
        temperature,
        normalize_dimension=(similarity_type == "l2"),
    )
    logits_rh, labels_rh = tcc_align_sequence_pair(
        r_seq[idx],
        h_seq[idx],
        similarity_type,
        temperature,
        normalize_dimension=(similarity_type == "l2"),
    )
    logits_list.extend([logits_hr, logits_rh])
    labels_list.extend([labels_hr, labels_rh])

  logits = torch.cat(logits_list, dim=0)
  labels = torch.cat(labels_list, dim=0)
  if loss_type == "classification":
    loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
  else:
    loss = tcc_regression_loss(
        logits,
        labels,
        loss_type,
        variance_lambda,
        huber_delta,
    )

  top1 = (logits.argmax(dim=1) == labels).float().mean()
  distances = torch.cdist(h_seq, r_seq).pow(2)
  pos = distances.diagonal(dim1=1, dim2=2).mean()
  off = (
      distances.sum() - distances.diagonal(dim1=1, dim2=2).sum()
  ) / max(1, distances.numel() - batch_size * num_timestamps)
  return loss, {
      "distances": distances,
      "top1": top1,
      "pos_dist": pos,
      "off_dist": off,
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


def compute_soft_temporal_infonce(
    z_a,
    z_b,
    num_sequences,
    num_timestamps,
    temperature,
    alpha,
    tau,
):
  expected = num_sequences * num_timestamps
  if z_a.shape[0] != expected or z_b.shape[0] != expected:
    raise ValueError(
        f"Expected {expected} embeddings, got {z_a.shape[0]} and {z_b.shape[0]}.")
  if tau <= 0:
    raise ValueError("--mv-soft-temporal-tau must be > 0.")
  if alpha < 0 or alpha > 1:
    raise ValueError("--mv-soft-temporal-alpha must be in [0, 1].")

  logits = (z_a @ z_b.t()) / temperature
  sequence_ids = torch.arange(expected, device=z_a.device) // num_timestamps
  positions = torch.arange(expected, device=z_a.device) % num_timestamps
  same_sequence = sequence_ids[:, None] == sequence_ids[None]
  temporal_dist = (positions[:, None] - positions[None]).abs().float()
  temporal_target = torch.exp(-temporal_dist / tau)
  temporal_target = temporal_target.masked_fill(~same_sequence, 0.0)
  temporal_target = temporal_target / temporal_target.sum(dim=1, keepdim=True)
  hard_target = torch.eye(expected, device=z_a.device, dtype=z_a.dtype)
  target = (1.0 - alpha) * hard_target + alpha * temporal_target.to(z_a.dtype)

  loss_ab = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
  loss_ba = -(target * F.log_softmax(logits.t(), dim=1)).sum(dim=1).mean()
  loss = 0.5 * (loss_ab + loss_ba)

  sims = z_a @ z_b.t()
  labels = torch.arange(expected, device=z_a.device)
  top1_ab = (sims.argmax(dim=1) == labels).float().mean()
  top1_ba = (sims.argmax(dim=0) == labels).float().mean()
  diag = sims.diag().mean()
  off = (sims.sum() - sims.diag().sum()) / max(1, sims.numel() - expected)
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


def init_distributed(args: argparse.Namespace):
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  distributed = world_size > 1
  if distributed:
    if not torch.cuda.is_available():
      raise RuntimeError("DDP training requires CUDA.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
  else:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool):
  if distributed and dist.is_initialized():
    dist.destroy_process_group()


def main() -> None:
  args = parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  distributed, rank, local_rank, world_size, device = init_distributed(args)
  is_main = rank == 0
  seed = args.seed + rank
  random.seed(seed)
  torch.manual_seed(seed)
  rng = random.Random(seed)

  max_groups = normalize_max_groups(getattr(args, "max_groups", None))
  args.training_index = getattr(args, "training_index", None)
  args.timestamp_groups = getattr(args, "timestamp_groups", None)
  if args.training_index is None and args.timestamp_groups is None:
    raise ValueError("Config must provide training_index or timestamp_groups.")
  if args.training_index is not None:
    args.camera_matched_pairing = True
  if args.camera_matched_pairing and args.num_multi_view <= 0:
    raise ValueError(
        "--camera-matched-pairing requires --num-multi-view > 0")
  if args.training_index is not None:
    paired_tracks = load_matched_training_index(
        args.training_index,
        min_groups=args.num_timestamps,
        combo_size=args.num_multi_view,
    )
  else:
    paired_tracks = load_tracks(
        args.timestamp_groups,
        args.num_timestamps,
        args.num_multi_view,
        max_groups,
        camera_matched_pairing=args.camera_matched_pairing,
        combo_size=args.num_multi_view,
    )
  episode_ids = sorted(paired_tracks, key=lambda x: int(x))
  task_to_episodes = build_task_to_episodes(paired_tracks)
  if len(episode_ids) < args.batch_episode_pairs:
    raise RuntimeError("Not enough paired episode tracks.")
  if (
      args.fusion_mode == "fixed_slot"
      and (not hasattr(args, "num_camera_slots") or args.num_camera_slots is None)
  ):
    args.num_camera_slots = infer_num_camera_slots(paired_tracks)
  if is_main:
    num_camera_slots = getattr(args, "num_camera_slots", None)
    print(
        f"loaded paired_episodes={len(episode_ids)} "
        f"tasks={len(task_to_episodes)} "
        f"num_timestamps={args.num_timestamps} "
        f"num_multi_view={args.num_multi_view} "
        f"fusion_mode={args.fusion_mode} "
        f"backbone={args.backbone} "
        f"softdtw_mode={args.softdtw_mode} "
        f"soft_alignment_temperature={args.soft_alignment_temperature} "
        f"soft_alignment_epsilon={args.soft_alignment_epsilon} "
        f"soft_alignment_rho={args.soft_alignment_rho} "
        f"soft_alignment_sinkhorn_iters={args.soft_alignment_sinkhorn_iters} "
        f"soft_alignment_struct_lambda={args.soft_alignment_struct_lambda} "
        f"soft_alignment_max_forward_step={args.soft_alignment_max_forward_step} "
        f"mv_soft_temporal={args.mv_soft_temporal} "
        f"mv_soft_temporal_alpha={args.mv_soft_temporal_alpha} "
        f"mv_soft_temporal_tau={args.mv_soft_temporal_tau} "
        f"pixel_aug={args.pixel_aug} "
        f"view_mask_mode={args.view_mask_mode} "
        f"view_mask_prob={args.view_mask_prob} "
        f"view_mask_min_views="
        f"{'auto_drop_one' if args.view_mask_mode == 'drop_one' else args.view_mask_min_views} "
        f"train_backbone_adapters={args.train_backbone_adapters} "
        f"num_camera_slots={num_camera_slots} "
        f"camera_matched_pairing={args.camera_matched_pairing} "
        f"world_size={world_size} "
        f"training_index={args.training_index}")

  transform = make_transform(args.image_size)
  aug_transform = (
      make_weak_aug_transform(args.image_size)
      if args.pixel_aug else transform
  )
  if args.fusion_mode == "fixed_slot":
    raw_model = FixedSlotFusionSoftDTW(
        num_camera_slots=args.num_camera_slots,
        embedding_size=args.embedding_size,
        fusion_size=args.fusion_size,
        pretrain_path=args.pretrain_path,
        backbone=args.backbone,
        train_layernorm=True,
        train_adapters=args.train_backbone_adapters,
    ).to(device).train()
  elif args.fusion_mode == "attention_pool":
    raw_model = ViewSetAttentionSoftDTW(
        embedding_size=args.embedding_size,
        fusion_size=args.fusion_size,
        pretrain_path=args.pretrain_path,
        backbone=args.backbone,
        train_layernorm=True,
        train_adapters=args.train_backbone_adapters,
        view_token_dropout=args.view_token_dropout,
        view_token_noise_std=args.view_token_noise_std,
        attention_dropout=args.attention_dropout,
        projector_dropout=args.projector_dropout,
    ).to(device).train()
  else:
    raise ValueError(f"Unknown fusion_mode: {args.fusion_mode}")
  if is_main:
    print(trainable_summary(raw_model))
  if distributed:
    model = DistributedDataParallel(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
  else:
    model = raw_model

  optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad],
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
  scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

  csv_name = "losses.csv" if is_main else f"losses_rank{rank}.csv"
  csv_path = args.out_dir / csv_name
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step",
        "loss_total",
        "loss_softdtw",
        "loss_aux",
        "loss_aux_h",
        "loss_aux_r",
        "loss_align",
        "loss_struct",
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
        "batch_episode_pairs",
        "images",
        "seconds",
        "cuda_mem_mb",
    ])
    f.flush()

    start = time.time()
    for step in range(1, args.max_iters + 1):
      iter_start = time.time()
      batch_episode_ids = sample_episode_batch(
          episode_ids,
          task_to_episodes,
          args.batch_episode_pairs,
          True,
          rng,
      )
      human_sequences = []
      robot_sequences = []
      for eid in batch_episode_ids:
        if args.camera_matched_pairing:
          combo_rec = rng.choice(paired_tracks[eid]["shared_combos"])
          if isinstance(combo_rec, dict):
            if "h" in combo_rec:
              h_groups = combo_rec["h"]
              r_groups = combo_rec["r"]
            else:
              combo = combo_rec["camera_ids"]
              h_groups = materialize_combo_groups(
                  paired_tracks[eid]["h_group_pool"],
                  combo_rec["h_group_indices"],
                  combo,
                  eid,
                  paired_tracks[eid]["task_id"],
                  "h",
              )
              r_groups = materialize_combo_groups(
                  paired_tracks[eid]["r_group_pool"],
                  combo_rec["r_group_indices"],
                  combo,
                  eid,
                  paired_tracks[eid]["task_id"],
                  "r",
              )
          else:
            h_groups = filter_groups_to_camera_combo(paired_tracks[eid]["h"], combo_rec)
            r_groups = filter_groups_to_camera_combo(paired_tracks[eid]["r"], combo_rec)
        else:
          h_groups = paired_tracks[eid]["h"]
          r_groups = paired_tracks[eid]["r"]
        human_sequences.append(
            stratified_group_sample(h_groups, args.num_timestamps, rng))
        robot_sequences.append(
            stratified_group_sample(r_groups, args.num_timestamps, rng))
      all_sequences = human_sequences + robot_sequences
      if args.fusion_mode == "fixed_slot":
        subset_sequences = make_view_dropout_sequences(
            all_sequences,
            args.num_multi_view,
            keep_ratio=0.75,
            disjoint=True,
            rng=rng,
        )
        images, group_idx, subset_idx, camera_ids, num_groups = prepare_subset_batch_images(
            args.data_root, subset_sequences, transform, device)
      elif args.fusion_mode == "attention_pool":
        if args.pixel_aug:
          (
              images,
              group_idx,
              camera_ids,
              aug_images,
              aug_group_idx,
              aug_camera_ids,
              num_groups,
          ) = prepare_clean_aug_batch_images(
              args.data_root,
              all_sequences,
              transform,
              aug_transform,
              device,
              args.view_mask_mode,
              args.view_mask_prob,
              args.view_mask_min_views,
              rng,
          )
        else:
          images, group_idx, camera_ids, num_groups = prepare_batch_images(
              args.data_root, all_sequences, transform, device)
          aug_images = None
          aug_group_idx = None
          aug_camera_ids = None
      else:
        raise ValueError(f"Unknown fusion_mode: {args.fusion_mode}")
      image_count = images.shape[0]
      if args.fusion_mode == "attention_pool" and aug_images is not None:
        image_count += aug_images.shape[0]

      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast(
          "cuda", enabled=args.amp and device.type == "cuda"):
        if args.fusion_mode == "fixed_slot":
          z_groups, _, z_subsets_aux = model(
              images, group_idx, subset_idx, camera_ids, num_groups)
        else:
          z_groups, z_aux_a, z_aux_b = model(
              images,
              group_idx,
              camera_ids,
              num_groups,
              aug_images=aug_images,
              aug_group_indices=aug_group_idx,
              aug_camera_ids=aug_camera_ids,
              aug_num_groups=num_groups,
              feature_view_mask_mode=(
                  args.view_mask_mode if not args.pixel_aug else "none"),
              feature_view_mask_prob=args.view_mask_prob,
              feature_view_mask_min_views=args.view_mask_min_views,
          )
        z = z_groups.reshape(
            2 * args.batch_episode_pairs, args.num_timestamps, -1)
        h_seq = z[:args.batch_episode_pairs]
        r_seq = z[args.batch_episode_pairs:]
        if args.softdtw_mode == "contrastive":
          loss_softdtw, metrics = compute_softdtw_contrastive(
              h_seq,
              r_seq,
              gamma=args.gamma,
              temperature=args.temperature,
              softdtw_divergence=args.softdtw_divergence,
          )
        elif args.softdtw_mode == "paired":
          loss_softdtw, metrics = compute_softdtw_paired(
              h_seq,
              r_seq,
              gamma=args.gamma,
              softdtw_divergence=args.softdtw_divergence,
          )
        elif args.softdtw_mode == "tcc":
          loss_softdtw, metrics = compute_tcc_cycleback(
              h_seq,
              r_seq,
              temperature=args.temperature,
              loss_type=args.tcc_loss_type,
              similarity_type=args.tcc_similarity_type,
              label_smoothing=args.tcc_label_smoothing,
              variance_lambda=args.tcc_variance_lambda,
              huber_delta=args.tcc_huber_delta,
          )
        elif args.softdtw_mode == "soft_alignment":
          loss_softdtw, metrics = compute_soft_alignment_paired(
              h_seq,
              r_seq,
              temperature=args.soft_alignment_temperature,
              epsilon=args.soft_alignment_epsilon,
              rho=args.soft_alignment_rho,
              sinkhorn_iters=args.soft_alignment_sinkhorn_iters,
              struct_lambda=args.soft_alignment_struct_lambda,
              max_forward_step=args.soft_alignment_max_forward_step,
          )
        else:
          raise ValueError(f"Unknown softdtw_mode: {args.softdtw_mode}")
        if args.fusion_mode == "fixed_slot":
          z_sub = z_subsets_aux.reshape(
              2 * args.batch_episode_pairs, args.num_timestamps, 2, -1)
          h_sub = z_sub[:args.batch_episode_pairs].reshape(
              args.batch_episode_pairs * args.num_timestamps, 2, -1)
          r_sub = z_sub[args.batch_episode_pairs:].reshape(
              args.batch_episode_pairs * args.num_timestamps, 2, -1)
          if args.mv_soft_temporal:
            loss_aux_h, aux_h = compute_soft_temporal_infonce(
                h_sub[:, 0],
                h_sub[:, 1],
                args.batch_episode_pairs,
                args.num_timestamps,
                args.mv_temperature,
                args.mv_soft_temporal_alpha,
                args.mv_soft_temporal_tau,
            )
            loss_aux_r, aux_r = compute_soft_temporal_infonce(
                r_sub[:, 0],
                r_sub[:, 1],
                args.batch_episode_pairs,
                args.num_timestamps,
                args.mv_temperature,
                args.mv_soft_temporal_alpha,
                args.mv_soft_temporal_tau,
            )
          else:
            loss_aux_h, aux_h = compute_multiview_infonce(
                h_sub[:, 0], h_sub[:, 1], args.mv_temperature)
            loss_aux_r, aux_r = compute_multiview_infonce(
                r_sub[:, 0], r_sub[:, 1], args.mv_temperature)
        else:
          z_aux_a = z_aux_a.reshape(
              2 * args.batch_episode_pairs, args.num_timestamps, -1)
          z_aux_b = z_aux_b.reshape(
              2 * args.batch_episode_pairs, args.num_timestamps, -1)
          h_aux_a = z_aux_a[:args.batch_episode_pairs].reshape(
              args.batch_episode_pairs * args.num_timestamps, -1)
          h_aux_b = z_aux_b[:args.batch_episode_pairs].reshape(
              args.batch_episode_pairs * args.num_timestamps, -1)
          r_aux_a = z_aux_a[args.batch_episode_pairs:].reshape(
              args.batch_episode_pairs * args.num_timestamps, -1)
          r_aux_b = z_aux_b[args.batch_episode_pairs:].reshape(
              args.batch_episode_pairs * args.num_timestamps, -1)
          if args.mv_soft_temporal:
            loss_aux_h, aux_h = compute_soft_temporal_infonce(
                h_aux_a,
                h_aux_b,
                args.batch_episode_pairs,
                args.num_timestamps,
                args.mv_temperature,
                args.mv_soft_temporal_alpha,
                args.mv_soft_temporal_tau,
            )
            loss_aux_r, aux_r = compute_soft_temporal_infonce(
                r_aux_a,
                r_aux_b,
                args.batch_episode_pairs,
                args.num_timestamps,
                args.mv_temperature,
                args.mv_soft_temporal_alpha,
                args.mv_soft_temporal_tau,
            )
          else:
            loss_aux_h, aux_h = compute_multiview_infonce(
                h_aux_a, h_aux_b, args.mv_temperature)
            loss_aux_r, aux_r = compute_multiview_infonce(
                r_aux_a, r_aux_b, args.mv_temperature)
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
      align_loss = metrics.get("align_loss", loss_softdtw)
      struct_loss = metrics.get(
          "struct_loss", loss_softdtw.new_zeros(()))
      row = [
          step,
          f"{loss.item():.8f}",
          f"{loss_softdtw.item():.8f}",
          f"{loss_aux.item():.8f}",
          f"{loss_aux_h.item():.8f}",
          f"{loss_aux_r.item():.8f}",
          f"{align_loss.item():.8f}",
          f"{struct_loss.item():.8f}",
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
          args.batch_episode_pairs,
          image_count,
          f"{seconds:.4f}",
          f"{mem_mb:.1f}",
      ]
      writer.writerow(row)
      if is_main and (step == 1 or step % args.log_every == 0):
        print(
            f"step {step:05d} total={row[1]} sdtw={row[2]} aux={row[3]} "
            f"align={row[6]} struct={row[7]} "
            f"sdtw_top1={row[8]} pos/off={row[9]}/{row[10]} "
            f"aux_top1_h/r={row[11]}/{row[12]} "
            f"aux_diag_off_h={row[13]}/{row[14]} "
            f"aux_diag_off_r={row[15]}/{row[16]} "
            f"emb_std={row[17]} episode_pairs={args.batch_episode_pairs} "
            f"imgs={image_count} "
            f"mem={row[21]}MB sec={row[20]}",
            flush=True,
        )
        f.flush()
      if is_main and args.save_every and step % args.save_every == 0:
        torch.save(
            {
                "step": step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            args.out_dir / f"checkpoint_{step:06d}.pt",
        )
    if is_main:
      print(f"done in {time.time() - start:.1f}s; losses={csv_path}")
  cleanup_distributed(distributed)


if __name__ == "__main__":
  main()
