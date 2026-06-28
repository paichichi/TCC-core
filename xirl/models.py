"""Backbone models used by the RH20T multi-view trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class HRAlignLateAdapter(nn.Module):
  """Late grouped 1x1 adapter used by the released AdaptedR3M checkpoint."""

  def __init__(
      self,
      channels: int = 2048,
      hidden_channels: int = 512,
      groups: int = 8,
  ):
    super().__init__()
    self.D_fc1 = nn.Conv2d(
        channels,
        hidden_channels,
        kernel_size=1,
        groups=groups,
    )
    self.D_mapping = nn.Conv2d(
        hidden_channels,
        hidden_channels,
        kernel_size=1,
    )
    self.D_fc2 = nn.Conv2d(
        hidden_channels,
        channels,
        kernel_size=1,
        groups=groups,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.relu(self.D_fc1(x), inplace=True)
    x = F.relu(self.D_mapping(x), inplace=True)
    return self.D_fc2(x)


class HRAlignR3MBackbone(nn.Module):
  """ResNet50/R3M backbone with HR-Align-style late adapters.

  The public HR-Align code is not available, so this mirrors the structure
  implied by AdaptedR3M.pyth:
    - convnet.* ResNet50 weights
    - convnet.late_adapter_{1,2,3}.D_fc{1,2}/D_mapping
    - MODEL.ADAPTER: late.layer.3.k.1.down.4.g.8

  We insert one residual adapter after each layer4 bottleneck block.
  """

  output_dim = 2048

  def __init__(
      self,
      pretrain_path: str = "",
      train_norm_affine: bool = True,
      train_adapters: bool = False,
      frozen_bn_stats: bool = True,
  ):
    super().__init__()
    self.frozen_bn_stats = frozen_bn_stats
    self.convnet = models.resnet50(weights=None)
    self.convnet.fc = nn.Identity()
    self.convnet.late_adapter_1 = HRAlignLateAdapter()
    self.convnet.late_adapter_2 = HRAlignLateAdapter()
    self.convnet.late_adapter_3 = HRAlignLateAdapter()
    self._zero_init_late_adapter_outputs()

    if pretrain_path:
      self.load_pretrained(pretrain_path)

    for param in self.parameters():
      param.requires_grad = False

    if train_norm_affine:
      for module in self.modules():
        if isinstance(module, nn.BatchNorm2d):
          if module.weight is not None:
            module.weight.requires_grad = True
          if module.bias is not None:
            module.bias.requires_grad = True

    if train_adapters:
      for name, param in self.named_parameters():
        if ".late_adapter_" in name:
          param.requires_grad = True

    if self.frozen_bn_stats:
      self._set_bn_eval()

  def _set_bn_eval(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.BatchNorm2d):
        module.eval()

  def _zero_init_late_adapter_outputs(self) -> None:
    for adapter in (
        self.convnet.late_adapter_1,
        self.convnet.late_adapter_2,
        self.convnet.late_adapter_3,
    ):
      nn.init.zeros_(adapter.D_fc2.weight)
      nn.init.zeros_(adapter.D_fc2.bias)

  def train(self, mode: bool = True):
    super().train(mode)
    if self.frozen_bn_stats:
      self._set_bn_eval()
    return self

  def load_pretrained(self, checkpoint_path: str):
    try:
      checkpoint = torch.load(
          checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
      checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model_state"), dict):
      source_state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("r3m"), dict):
      source_state = checkpoint["r3m"]
    else:
      source_state = _unwrap_state_dict(checkpoint)

    target_state = self.state_dict()
    mapped_state = {}
    skipped = []
    for key, value in source_state.items():
      candidates = [
          key,
          key.removeprefix("module."),
          key.removeprefix("module.convnet."),
          f"convnet.{key}",
          f"convnet.{key.removeprefix('module.convnet.')}",
      ]
      mapped_key = next(
          (candidate for candidate in candidates if candidate in target_state),
          None,
      )
      if mapped_key in target_state and target_state[mapped_key].shape == value.shape:
        mapped_state[mapped_key] = value
      else:
        skipped.append(key)

    missing, unexpected = self.load_state_dict(mapped_state, strict=False)
    print(
        "HRAlignR3MBackbone loaded "
        f"{len(mapped_state)} tensors from {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"skipped={len(skipped)}")
    if missing:
      print(f"HRAlignR3MBackbone missing (first 20): {missing[:20]}")
    if skipped:
      print(f"HRAlignR3MBackbone skipped source keys (first 20): {skipped[:20]}")

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    x = self.convnet.conv1(images)
    x = self.convnet.bn1(x)
    x = self.convnet.relu(x)
    x = self.convnet.maxpool(x)

    x = self.convnet.layer1(x)
    x = self.convnet.layer2(x)
    x = self.convnet.layer3(x)

    x = self.convnet.layer4[0](x)
    x = x + self.convnet.late_adapter_1(x)
    x = self.convnet.layer4[1](x)
    x = x + self.convnet.late_adapter_2(x)
    x = self.convnet.layer4[2](x)
    x = x + self.convnet.late_adapter_3(x)

    x = self.convnet.avgpool(x)
    return torch.flatten(x, 1)


class R3MResNet50Backbone(nn.Module):
  """Original R3M ResNet50 visual backbone without HR-Align adapters."""

  output_dim = 2048

  def __init__(
      self,
      pretrain_path: str = "",
      train_norm_affine: bool = True,
      frozen_bn_stats: bool = True,
  ):
    super().__init__()
    self.frozen_bn_stats = frozen_bn_stats
    self.convnet = models.resnet50(weights=None)
    self.convnet.fc = nn.Identity()

    if pretrain_path:
      self.load_pretrained(pretrain_path)

    for param in self.parameters():
      param.requires_grad = False

    if train_norm_affine:
      for module in self.modules():
        if isinstance(module, nn.BatchNorm2d):
          if module.weight is not None:
            module.weight.requires_grad = True
          if module.bias is not None:
            module.bias.requires_grad = True

    if self.frozen_bn_stats:
      self._set_bn_eval()

  def _set_bn_eval(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.BatchNorm2d):
        module.eval()

  def train(self, mode: bool = True):
    super().train(mode)
    if self.frozen_bn_stats:
      self._set_bn_eval()
    return self

  def load_pretrained(self, checkpoint_path: str):
    try:
      checkpoint = torch.load(
          checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
      checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("r3m"), dict):
      source_state = checkpoint["r3m"]
    else:
      source_state = _unwrap_state_dict(checkpoint)

    target_state = self.state_dict()
    mapped_state = {}
    skipped = []
    for key, value in source_state.items():
      candidates = [
          key,
          key.removeprefix("module."),
          key.removeprefix("module.convnet."),
          key.removeprefix("convnet."),
          f"convnet.{key.removeprefix('module.convnet.')}",
      ]
      mapped_key = next(
          (candidate for candidate in candidates if candidate in target_state),
          None,
      )
      if mapped_key in target_state and target_state[mapped_key].shape == value.shape:
        mapped_state[mapped_key] = value
      else:
        skipped.append(key)

    missing, unexpected = self.load_state_dict(mapped_state, strict=False)
    print(
        "R3MResNet50Backbone loaded "
        f"{len(mapped_state)} tensors from {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"skipped={len(skipped)}")
    if missing:
      print(f"R3MResNet50Backbone missing (first 20): {missing[:20]}")
    if skipped:
      print(f"R3MResNet50Backbone skipped source keys (first 20): {skipped[:20]}")

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    return self.convnet(images)


def build_backbone(
    backbone: str,
    pretrain_path: str = "",
    train_norm_affine: bool = True,
    train_adapters: bool = False,
) -> nn.Module:
  if backbone in ("vit_b16", "vit"):
    model = ViTB16Backbone(
        pretrain_path=pretrain_path,
        vit_weights="none",
        pooling="patch_mean",
    )
    for param in model.parameters():
      param.requires_grad = False
    if train_norm_affine:
      for module in model.modules():
        if isinstance(module, nn.LayerNorm):
          for param in module.parameters():
            param.requires_grad = True
    return model
  if backbone in ("r3m_align_l", "hralign_r3m_l", "adapted_r3m"):
    return HRAlignR3MBackbone(
        pretrain_path=pretrain_path,
        train_norm_affine=train_norm_affine,
        train_adapters=train_adapters,
        frozen_bn_stats=True,
    )
  if backbone in ("r3m_late_adapter", "r3m_adapter", "r3m_hralign_style"):
    return HRAlignR3MBackbone(
        pretrain_path=pretrain_path,
        train_norm_affine=train_norm_affine,
        train_adapters=train_adapters,
        frozen_bn_stats=True,
    )
  if backbone in ("r3m_resnet50", "r3m", "unadapted_r3m"):
    if train_adapters:
      raise ValueError(
          "train_adapters=True is only valid for HR-Align adapted R3M.")
    return R3MResNet50Backbone(
        pretrain_path=pretrain_path,
        train_norm_affine=train_norm_affine,
        frozen_bn_stats=True,
    )
  raise ValueError(f"Unsupported backbone: {backbone}")
