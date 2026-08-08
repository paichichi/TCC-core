"""Backbone models used by the RH20T multi-view trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


R3M_LATE_ADAPTER_LAYOUT = "post_layer4_sequential_v1"
R3M_ADAPTER_INIT_RELEASE = "release_zero_mapping"
R3M_ADAPTER_INIT_TRAINABLE = "trainable_identity"


class ReferenceAffineBatchNorm2d(nn.BatchNorm2d):
  """BatchNorm with an immutable copy of its pretrained affine parameters."""

  def __init__(self, source: nn.BatchNorm2d):
    if not source.affine:
      raise ValueError("BN-affine tuning requires affine BatchNorm layers.")
    super().__init__(
        source.num_features,
        eps=source.eps,
        momentum=source.momentum,
        affine=True,
        track_running_stats=source.track_running_stats,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    self.load_state_dict(source.state_dict())
    self.register_buffer(
        "reference_weight", source.weight.detach().clone(), persistent=False)
    self.register_buffer(
        "reference_bias", source.bias.detach().clone(), persistent=False)
    self.use_reference_affine = False

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    weight = self.reference_weight if self.use_reference_affine else self.weight
    bias = self.reference_bias if self.use_reference_affine else self.bias
    return F.batch_norm(
        inputs,
        self.running_mean,
        self.running_var,
        weight,
        bias,
        training=False,
        momentum=0.0,
        eps=self.eps,
    )


def _install_reference_bn_affines(module: nn.Module) -> None:
  for name, child in list(module.named_children()):
    if isinstance(child, ReferenceAffineBatchNorm2d):
      continue
    if isinstance(child, nn.BatchNorm2d):
      setattr(module, name, ReferenceAffineBatchNorm2d(child))
    else:
      _install_reference_bn_affines(child)


def validate_r3m_adapter_layout(
    model_format: dict,
    allow_unversioned: bool,
) -> None:
  checkpoint_layout = model_format.get("r3m_late_adapter_layout")
  if checkpoint_layout is None:
    if allow_unversioned:
      return
    raise ValueError(
        "Refusing an unversioned R3M adapter checkpoint: "
        f"expected layout={R3M_LATE_ADAPTER_LAYOUT!r}.")
  if checkpoint_layout != R3M_LATE_ADAPTER_LAYOUT:
    raise ValueError(
        "Refusing an incompatible R3M adapter checkpoint: "
        f"expected layout={R3M_LATE_ADAPTER_LAYOUT!r}, "
        f"got {checkpoint_layout!r}.")


def validate_r3m_adapter_init(
    model_format: dict,
    expected_init: str,
    allow_unversioned: bool,
) -> None:
  checkpoint_init = model_format.get("r3m_adapter_init")
  if checkpoint_init is None:
    if allow_unversioned:
      return
    raise ValueError(
        "Refusing an R3M adapter checkpoint without initialization metadata: "
        f"expected init={expected_init!r}.")
  if checkpoint_init != expected_init:
    raise ValueError(
        "Refusing an R3M adapter checkpoint with incompatible initialization: "
        f"expected init={expected_init!r}, got {checkpoint_init!r}.")


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
  """Late grouped 1x1 residual adapter used by HR-Align and Method 3."""

  def __init__(
      self,
      channels: int = 2048,
      hidden_channels: int = 512,
      groups: int = 8,
      init_mode: str = R3M_ADAPTER_INIT_RELEASE,
  ):
    super().__init__()
    if init_mode not in (
        R3M_ADAPTER_INIT_RELEASE,
        R3M_ADAPTER_INIT_TRAINABLE,
    ):
      raise ValueError(f"Unsupported R3M adapter init mode: {init_mode}")
    self.init_mode = init_mode
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
    self.act = nn.ReLU()

    nn.init.zeros_(self.D_fc1.bias)
    nn.init.zeros_(self.D_mapping.bias)
    nn.init.zeros_(self.D_fc2.bias)
    if init_mode == R3M_ADAPTER_INIT_RELEASE:
      # Compatibility with the released HR-Align implementation/checkpoint.
      # ReLU at the all-zero mapping makes this initialization unsuitable for
      # training a fresh adapter, but it must remain available for baselines.
      nn.init.zeros_(self.D_mapping.weight)
    else:
      # A zero final projection makes the residual branch exactly zero while
      # preserving a live gradient path into the branch after the first step.
      nn.init.zeros_(self.D_fc2.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = self.act(self.D_fc1(x))
    residual = self.act(self.D_mapping(residual))
    residual = self.D_fc2(residual)
    return x + residual


class HRAlignR3MBackbone(nn.Module):
  """ResNet50/R3M backbone with release-compatible HR-Align late adapters.

  This follows the official downstream source and AdaptedR3M.pyth:
    - convnet.* ResNet50 weights
    - convnet.late_adapter_{1,2,3}.D_fc{1,2}/D_mapping
    - MODEL.ADAPTER: late.layer.3.k.1.down.4.g.8

  All three residual adapters run sequentially after the complete layer4.
  """

  output_dim = 2048
  adapter_layout = R3M_LATE_ADAPTER_LAYOUT

  def __init__(
      self,
      pretrain_path: str = "",
      train_norm_affine: bool = True,
      train_adapters: bool = False,
      frozen_bn_stats: bool = True,
      adapter_init: str = R3M_ADAPTER_INIT_RELEASE,
      allow_unversioned_adapter_checkpoint: bool = False,
  ):
    super().__init__()
    self.frozen_bn_stats = frozen_bn_stats
    self.adapter_init = adapter_init
    self.allow_unversioned_adapter_checkpoint = (
        allow_unversioned_adapter_checkpoint)
    self.convnet = models.resnet50(weights=None)
    self.convnet.fc = nn.Identity()
    self.convnet.late_adapter_1 = HRAlignLateAdapter(init_mode=adapter_init)
    self.convnet.late_adapter_2 = HRAlignLateAdapter(init_mode=adapter_init)
    self.convnet.late_adapter_3 = HRAlignLateAdapter(init_mode=adapter_init)

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
    adapter_keys = {
        key for key in target_state if ".late_adapter_" in key
    }
    mapped_state = {}
    skipped = []
    shape_mismatches = []
    for key, value in source_state.items():
      candidates = [
          key,
          key.removeprefix("module."),
          key.removeprefix("backbone."),
          key.removeprefix("module.backbone."),
          key.removeprefix("module.convnet."),
          key.removeprefix("backbone.convnet."),
          key.removeprefix("module.backbone.convnet."),
          f"convnet.{key}",
          f"convnet.{key.removeprefix('module.convnet.')}",
          f"convnet.{key.removeprefix('backbone.convnet.')}",
          f"convnet.{key.removeprefix('module.backbone.convnet.')}",
      ]
      mapped_key = next(
          (candidate for candidate in candidates if candidate in target_state),
          None,
      )
      if (
          mapped_key in target_state
          and isinstance(value, torch.Tensor)
          and target_state[mapped_key].shape == value.shape
      ):
        mapped_state[mapped_key] = value
      elif mapped_key in target_state and isinstance(value, torch.Tensor):
        shape_mismatches.append(
            (mapped_key, tuple(value.shape), tuple(target_state[mapped_key].shape)))
      else:
        skipped.append(key)

    if shape_mismatches:
      raise ValueError(
          "R3M checkpoint tensor shape mismatch (first 10): "
          f"{shape_mismatches[:10]}")
    loaded_keys = set(mapped_state)
    loaded_adapter_keys = loaded_keys & adapter_keys
    missing_keys = set(target_state) - loaded_keys
    if loaded_adapter_keys:
      if loaded_adapter_keys != adapter_keys:
        missing_adapters = sorted(adapter_keys - loaded_adapter_keys)
        raise ValueError(
            "R3M checkpoint contains an incomplete adapter: "
            f"missing={missing_adapters}")
      model_format = (
          checkpoint.get("model_format", {})
          if isinstance(checkpoint, dict) else {})
      validate_r3m_adapter_layout(
          model_format,
          allow_unversioned=self.allow_unversioned_adapter_checkpoint,
      )
      validate_r3m_adapter_init(
          model_format,
          expected_init=self.adapter_init,
          allow_unversioned=self.allow_unversioned_adapter_checkpoint,
      )
      if missing_keys:
        raise ValueError(
            "R3M adapted checkpoint is missing backbone tensors: "
            f"{sorted(missing_keys)[:20]}")
    elif missing_keys != adapter_keys:
      missing_base = sorted(missing_keys - adapter_keys)
      raise ValueError(
          "R3M base checkpoint did not initialize the complete ResNet50: "
          f"missing={missing_base[:20]}")

    missing, unexpected = self.load_state_dict(mapped_state, strict=False)
    expected_missing = sorted(adapter_keys if not loaded_adapter_keys else ())
    if sorted(missing) != expected_missing or unexpected:
      raise RuntimeError(
          "Unexpected R3M load result: "
          f"missing={missing}, unexpected={unexpected}")
    print(
        "HRAlignR3MBackbone loaded "
        f"{len(mapped_state)} tensors from {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"skipped={len(skipped)}")
    if missing:
      print(f"HRAlignR3MBackbone missing (first 20): {missing[:20]}")
    if skipped:
      print(f"HRAlignR3MBackbone skipped source keys (first 20): {skipped[:20]}")

  def forward_base(self, images: torch.Tensor) -> torch.Tensor:
    """Encode images through the frozen ResNet, stopping after layer4."""
    x = self.convnet.conv1(images)
    x = self.convnet.bn1(x)
    x = self.convnet.relu(x)
    x = self.convnet.maxpool(x)

    x = self.convnet.layer1(x)
    x = self.convnet.layer2(x)
    x = self.convnet.layer3(x)
    x = self.convnet.layer4(x)
    return x

  def apply_adapters(self, features: torch.Tensor) -> torch.Tensor:
    """Apply the three sequential post-layer4 residual adapters."""
    x = features
    x = self.convnet.late_adapter_1(x)
    x = self.convnet.late_adapter_2(x)
    x = self.convnet.late_adapter_3(x)
    return x

  def pool_features(self, features: torch.Tensor) -> torch.Tensor:
    return torch.flatten(self.convnet.avgpool(features), 1)

  def forward_unadapted(self, images: torch.Tensor) -> torch.Tensor:
    return self.pool_features(self.forward_base(images))

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    features = self.forward_base(images)
    return self.pool_features(self.apply_adapters(features))


class R3MResNet50Backbone(nn.Module):
  """Original R3M ResNet50 with controlled BN-affine tuning."""

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

    # Preserve the exact R3M affine function for the frozen control paths.
    _install_reference_bn_affines(self.convnet)

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

  def _set_reference_affine(self, enabled: bool) -> None:
    for module in self.modules():
      if isinstance(module, ReferenceAffineBatchNorm2d):
        module.use_reference_affine = enabled

  def _forward_feature_map(self, images: torch.Tensor) -> torch.Tensor:
    x = self.convnet.conv1(images)
    x = self.convnet.bn1(x)
    x = self.convnet.relu(x)
    x = self.convnet.maxpool(x)
    x = self.convnet.layer1(x)
    x = self.convnet.layer2(x)
    x = self.convnet.layer3(x)
    x = self.convnet.layer4(x)
    return x

  def forward_base(self, images: torch.Tensor) -> torch.Tensor:
    """Run the immutable pretrained BN-affine reference path."""
    self._set_reference_affine(True)
    try:
      return self._forward_feature_map(images)
    finally:
      self._set_reference_affine(False)

  def forward_adapted(self, images: torch.Tensor) -> torch.Tensor:
    """Run the trainable BN-affine path."""
    self._set_reference_affine(False)
    return self._forward_feature_map(images)

  def pool_features(self, features: torch.Tensor) -> torch.Tensor:
    return torch.flatten(self.convnet.avgpool(features), 1)

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
    return self.pool_features(self.forward_adapted(images))


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
        adapter_init=R3M_ADAPTER_INIT_RELEASE,
        allow_unversioned_adapter_checkpoint=True,
    )
  if backbone in ("r3m_late_adapter", "r3m_adapter", "r3m_hralign_style"):
    return HRAlignR3MBackbone(
        pretrain_path=pretrain_path,
        train_norm_affine=train_norm_affine,
        train_adapters=train_adapters,
        frozen_bn_stats=True,
        adapter_init=R3M_ADAPTER_INIT_TRAINABLE,
        allow_unversioned_adapter_checkpoint=False,
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
