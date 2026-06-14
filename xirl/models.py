"""Backbone models used by the RH20T multi-view trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def _unwrap_state_dict(checkpoint):
  if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
    return checkpoint["model"]
  if isinstance(checkpoint, dict) and isinstance(
      checkpoint.get("state_dict"), dict):
    return checkpoint["state_dict"]
  return checkpoint


def _mae_to_torchvision_vit_key(key: str) -> str | None:
  """Map MAE-style ViT checkpoint keys to torchvision ViT-B/16 keys."""
  direct = {
      "cls_token": "class_token",
      "pos_embed": "encoder.pos_embedding",
      "patch_embed.proj.weight": "conv_proj.weight",
      "patch_embed.proj.bias": "conv_proj.bias",
      "norm.weight": "encoder.ln.weight",
      "norm.bias": "encoder.ln.bias",
  }
  if key in direct:
    return direct[key]

  parts = key.split(".")
  if len(parts) < 4 or parts[0] != "blocks":
    return None

  block = f"encoder.layers.encoder_layer_{parts[1]}"
  suffix = ".".join(parts[2:])
  mapping = {
      "norm1.weight": "ln_1.weight",
      "norm1.bias": "ln_1.bias",
      "attn.qkv.weight": "self_attention.in_proj_weight",
      "attn.qkv.bias": "self_attention.in_proj_bias",
      "attn.proj.weight": "self_attention.out_proj.weight",
      "attn.proj.bias": "self_attention.out_proj.bias",
      "norm2.weight": "ln_2.weight",
      "norm2.bias": "ln_2.bias",
      "mlp.fc1.weight": "mlp.0.weight",
      "mlp.fc1.bias": "mlp.0.bias",
      "mlp.fc2.weight": "mlp.3.weight",
      "mlp.fc2.bias": "mlp.3.bias",
  }
  mapped_suffix = mapping.get(suffix)
  if mapped_suffix is None:
    return None
  return f"{block}.{mapped_suffix}"


class ViTB16Backbone(nn.Module):
  """Torchvision ViT-B/16 backbone with optional checkpoint loading."""

  output_dim = 768

  def __init__(
      self,
      pretrain_path: str = "",
      vit_weights: str = "imagenet",
      pooling: str = "cls",
  ):
    super().__init__()
    if pooling not in ["cls", "patch_mean"]:
      raise ValueError(f"Unsupported ViT pooling: {pooling}")
    self.pooling = pooling

    weights = None
    if vit_weights == "imagenet" and not pretrain_path:
      weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    elif vit_weights not in ["imagenet", "none", ""]:
      raise ValueError(f"Unsupported ViT weights: {vit_weights}")

    self.model = models.vit_b_16(weights=weights)
    self.model.heads = nn.Identity()
    if pretrain_path:
      self.load_pretrained(pretrain_path)

  def load_pretrained(self, checkpoint_path: str):
    try:
      checkpoint = torch.load(
          checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
      checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = _unwrap_state_dict(checkpoint)
    target_state = self.model.state_dict()

    mapped_state = {}
    skipped = []
    for key, value in source_state.items():
      candidates = [
          key,
          key.removeprefix("module."),
          key.removeprefix("backbone."),
          key.removeprefix("backbone.model."),
      ]
      mapped_key = next(
          (candidate for candidate in candidates if candidate in target_state),
          None,
      )
      if mapped_key is None:
        mapped_key = _mae_to_torchvision_vit_key(key)
      if mapped_key in target_state and target_state[mapped_key].shape == value.shape:
        mapped_state[mapped_key] = value
      else:
        skipped.append(key)

    missing, unexpected = self.model.load_state_dict(mapped_state, strict=False)
    print(
        "ViTB16Backbone loaded "
        f"{len(mapped_state)} tensors from {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"skipped={len(skipped)}")
    if missing:
      print(f"ViTB16Backbone missing (first 20): {missing[:20]}")
    if skipped:
      print(f"ViTB16Backbone skipped source keys (first 20): {skipped[:20]}")

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    if self.pooling == "cls":
      return self.model(images)

    tokens = self.model._process_input(images)  # pylint: disable=protected-access
    batch_size = tokens.shape[0]
    class_token = self.model.class_token.expand(batch_size, -1, -1)
    tokens = torch.cat([class_token, tokens], dim=1)
    tokens = self.model.encoder(tokens)
    return tokens[:, 1:].mean(dim=1)
