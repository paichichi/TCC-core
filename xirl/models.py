# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self supervised models."""

import abc
import math
from typing import Union

import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


@dataclasses.dataclass
class SelfSupervisedOutput:
  """The output of a self-supervised model."""

  frames: Union[np.ndarray, torch.FloatTensor]
  feats: Union[np.ndarray, torch.FloatTensor]
  embs: Union[np.ndarray, torch.FloatTensor]

  def squeeze(self, dim):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      kwargs[k] = v.squeeze(dim)
    return self.__class__(**kwargs)

  def cpu(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      kwargs[k] = v.cpu()
    return self.__class__(**kwargs)

  def numpy(self):
    kwargs = {}
    for k, v in dataclasses.asdict(self).items():
      if k != "frames":
        kwargs[k] = v.cpu().detach().numpy()
    kwargs["frames"] = self.frames.permute(0, 2, 3, 1).cpu().detach().numpy()
    return self.__class__(**kwargs)

  @classmethod
  def merge(
      cls, output_list):
    kwargs = {}
    for k in dataclasses.asdict(output_list[0]).keys():
      kwargs[k] = torch.cat([getattr(o, k) for o in output_list], dim=1)
    return cls(**kwargs)


class SelfSupervisedModel(abc.ABC, nn.Module):
  """A self-supervised model trained on video data."""

  @abc.abstractmethod
  def __init__(
      self,
      num_ctx_frames,
      normalize_embeddings,
      learnable_temp,
  ):
    super().__init__()

    self.num_ctx_frames = num_ctx_frames
    self.normalize_embeddings = normalize_embeddings
    self.learnable_temp = learnable_temp

    # Log-parameterized multiplicative softmax temperature param.
    if learnable_temp:
      self.logit_scale = nn.Parameter(torch.ones([]))

  def forward(self, x):
    """Forward the video frames through the network.

    Args:
      x: The video frames of shape (B, T, C, H, W). If there are S video frames
        and we are using X context frames, then T = S * X.

    Returns:
      An instance of SelfSupervisedOutput.
    """
    batch_size, t, c, h, w = x.shape
    x_flat = x.view((batch_size * t, c, h, w))
    feats = self.backbone(x_flat)
    feats_flat = torch.flatten(feats, 1)
    embs = self.encoder(feats_flat)
    if self.normalize_embeddings:
      embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
    if self.learnable_temp:
      logit_scale = self.logit_scale.exp()
      embs = logit_scale * embs
    embs = embs.view((batch_size, t, -1))
    feats = feats.view((batch_size, t, -1))
    return SelfSupervisedOutput(frames=x, feats=feats, embs=embs)

  @torch.no_grad()
  def infer(
      self,
      x,
      max_batch_size = 128,
  ):
    """Forward at inference with possible very large batch sizes."""
    # Figure out a max batch size that's a multiple of the number of context
    # frames. This is so we can support large videos with many frames.
    lcm = self.num_ctx_frames
    effective_bs = math.floor(max_batch_size / lcm) * lcm
    if x.shape[1] > effective_bs:
      out = []
      for i in range(math.ceil(x.shape[1] / effective_bs)):
        sub_frames = x[:, i * effective_bs:(i + 1) * effective_bs]
        out.append(self.forward(sub_frames).cpu())
      out = SelfSupervisedOutput.merge(out)
    else:
      out = self.forward(x).cpu()
    return out.squeeze(0)


def _unwrap_state_dict(checkpoint):
  if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
    return checkpoint["model"]
  if isinstance(checkpoint, dict) and isinstance(
      checkpoint.get("state_dict"), dict):
    return checkpoint["state_dict"]
  return checkpoint


def _mae_to_torchvision_vit_key(key):
  if key == "cls_token":
    return "class_token"
  if key == "pos_embed":
    return "encoder.pos_embedding"
  if key == "patch_embed.proj.weight":
    return "conv_proj.weight"
  if key == "patch_embed.proj.bias":
    return "conv_proj.bias"
  if key == "norm.weight":
    return "encoder.ln.weight"
  if key == "norm.bias":
    return "encoder.ln.bias"

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
  """A torchvision ViT-B/16 backbone with optional checkpoint loading."""

  output_dim = 768

  def __init__(self, pretrain_path="", vit_weights="imagenet"):
    super().__init__()
    weights = None
    if vit_weights == "imagenet" and not pretrain_path:
      weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    elif vit_weights not in ["imagenet", "none", ""]:
      raise ValueError(f"Unsupported ViT weights: {vit_weights}")

    self.model = models.vit_b_16(weights=weights)
    self.model.heads = nn.Identity()
    if pretrain_path:
      self.load_pretrained(pretrain_path)

  def load_pretrained(self, checkpoint_path):
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
      mapped_key = None
      for candidate in candidates:
        if candidate in target_state:
          mapped_key = candidate
          break
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

  def forward(self, x):
    return self.model(x)


class ViTB16LinearEncoderNet(SelfSupervisedModel):
  """A ViT-B/16 backbone with a linear TCC embedding head."""

  def __init__(
      self,
      embedding_size,
      pretrain_path="",
      vit_weights="imagenet",
      trainable_scope="all",
      *args,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.backbone = ViTB16Backbone(pretrain_path, vit_weights)
    self.encoder = nn.Linear(self.backbone.output_dim, embedding_size)
    self.set_trainable_scope(trainable_scope)

  def set_trainable_scope(self, trainable_scope):
    if trainable_scope == "all":
      return

    for param in self.parameters():
      param.requires_grad = False

    if trainable_scope in ["head", "layernorm_head", "ln_head"]:
      for param in self.encoder.parameters():
        param.requires_grad = True
      if self.learnable_temp:
        self.logit_scale.requires_grad = True

    if trainable_scope in ["layernorm_head", "ln_head"]:
      for module in self.backbone.modules():
        if isinstance(module, nn.LayerNorm):
          for param in module.parameters():
            param.requires_grad = True
      return

    if trainable_scope != "head":
      raise ValueError(f"Unsupported ViT trainable scope: {trainable_scope}")


class Resnet18LinearEncoderNet(SelfSupervisedModel):
  """A resnet18 backbone with a linear encoder head."""

  def __init__(self, embedding_size, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Visual backbone.
    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    layers_ = list(resnet.children())[:-1]
    self.backbone = nn.Sequential(*layers_)

    # Encoder.
    self.encoder = nn.Linear(num_ftrs, embedding_size)
