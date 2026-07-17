from __future__ import annotations

from collections.abc import Mapping
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary checkpoint at {path}.")
    return checkpoint


def extract_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "r3m", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return dict(checkpoint)
    raise KeyError(
        "Checkpoint does not contain model_state, r3m, state_dict, or model."
    )


class ReleasedLateAdapter(nn.Module):
    """Adapter implemented by the official HumanRobotAlign downstream source.

    The public release uses three convolution layers even though Eq. (3) in the
    paper presents a two-layer bottleneck. D_mapping is zero-initialized exactly
    as in the release.
    """

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
        self.act = nn.ReLU()

        nn.init.zeros_(self.D_mapping.weight)
        nn.init.zeros_(self.D_mapping.bias)
        nn.init.zeros_(self.D_fc1.bias)
        nn.init.zeros_(self.D_fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.D_fc1(features))
        residual = self.act(self.D_mapping(residual))
        residual = self.D_fc2(residual)
        return features + residual


class R3MSpatialBackbone(nn.Module):
    """Torchvision ResNet-50 with optional release-compatible late adapters."""

    output_dim = 2048

    def __init__(self, with_late_adapters: bool):
        super().__init__()
        self.convnet = models.resnet50(weights=None)
        self.convnet.fc = nn.Identity()
        self.with_late_adapters = with_late_adapters
        if with_late_adapters:
            self.convnet.late_adapter_1 = ReleasedLateAdapter()
            self.convnet.late_adapter_2 = ReleasedLateAdapter()
            self.convnet.late_adapter_3 = ReleasedLateAdapter()

    def forward_base(self, images: torch.Tensor) -> torch.Tensor:
        x = self.convnet.conv1(images)
        x = self.convnet.bn1(x)
        x = self.convnet.relu(x)
        x = self.convnet.maxpool(x)
        x = self.convnet.layer1(x)
        x = self.convnet.layer2(x)
        x = self.convnet.layer3(x)
        return self.convnet.layer4(x)

    def apply_adapters(self, features: torch.Tensor) -> torch.Tensor:
        if not self.with_late_adapters:
            return features
        # The official source applies all three adapters sequentially after the
        # complete layer4, not after its individual bottleneck blocks.
        features = self.convnet.late_adapter_1(features)
        features = self.convnet.late_adapter_2(features)
        return self.convnet.late_adapter_3(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.apply_adapters(self.forward_base(images))


def _visual_state_from_source(
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in source_state.items():
        normalized = key.removeprefix("module.")
        if normalized.startswith("convnet."):
            result[normalized] = value
    return result


def _language_state_from_source(
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in source_state.items():
        normalized = key.removeprefix("module.")
        if normalized.startswith("lang_enc.model."):
            result[normalized] = value.detach().cpu().clone()
    return result


def _load_visual_state(
    backbone: R3MSpatialBackbone,
    visual_state: Mapping[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    target = backbone.state_dict()
    compatible = {
        key: value
        for key, value in visual_state.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = backbone.load_state_dict(compatible, strict=False)
    base_missing = [
        key for key in missing if ".late_adapter_" not in key
    ]
    if base_missing:
        raise RuntimeError(
            "R3M checkpoint is missing visual backbone tensors: "
            f"{base_missing[:20]}"
        )
    return list(missing), list(unexpected)


class HRAlignR3ML(nn.Module):
    """Three-stream R3M-Align-L model reconstructed from paper and release."""

    def __init__(
        self,
        pretrain_path: str | Path,
        adapted_bn_mode: str = "shared_stream_stats",
        normalize_language_query: bool = False,
        normalize_visual_tokens_for_attention: bool = False,
        normalize_pooled_features: bool = True,
    ):
        super().__init__()
        if adapted_bn_mode not in {
            "shared_stream_stats",
            "robot_stats",
            "frozen",
        }:
            raise ValueError(
                "adapted_bn_mode must be shared_stream_stats, robot_stats, "
                "or frozen."
            )
        self.adapted_bn_mode = adapted_bn_mode
        self.normalize_language_query = normalize_language_query
        self.normalize_visual_tokens_for_attention = (
            normalize_visual_tokens_for_attention
        )
        self.normalize_pooled_features = normalize_pooled_features

        checkpoint = load_torch_checkpoint(pretrain_path)
        source_state = extract_model_state(checkpoint)
        visual_state = _visual_state_from_source(source_state)
        if not visual_state:
            raise RuntimeError(
                f"No module.convnet.* R3M weights found in {pretrain_path}."
            )

        self.adapted = R3MSpatialBackbone(with_late_adapters=True)
        _load_visual_state(self.adapted, visual_state)

        self.lang_linear = nn.Linear(768, self.adapted.output_dim)
        self._source_language_state = _language_state_from_source(source_state)
        if len(self._source_language_state) != 100:
            raise RuntimeError(
                "Expected 100 DistilBERT tensors in the R3M checkpoint; "
                f"found {len(self._source_language_state)}."
            )

        for parameter in self.adapted.parameters():
            parameter.requires_grad = False
        for name, parameter in self.adapted.named_parameters():
            if ".late_adapter_" in name:
                parameter.requires_grad = True

        self.train(True)

    @property
    def source_language_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.clone()
            for key, value in self._source_language_state.items()
        }

    def _set_bn_training(self, enabled: bool) -> None:
        for module in self.adapted.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(enabled)

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_bn_training(
            mode and self.adapted_bn_mode == "shared_stream_stats"
        )
        return self

    def _forward_visual_streams(
        self,
        flat_human: torch.Tensor,
        flat_robot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the three paper streams through one shared R3M backbone.

        The released checkpoint's BatchNorm counters are consistent with one
        human and two robot base-backbone calls per pair batch. Keeping the
        calls separate also preserves the private run's inferred stream-wise
        BN updates instead of collapsing them into one concatenated call.
        """
        if self.adapted_bn_mode == "robot_stats":
            self._set_bn_training(False)
            human_reference = self.adapted.forward_base(flat_human)
            robot_reference = self.adapted.forward_base(flat_robot)
            self._set_bn_training(self.training)
            robot_adapted_base = self.adapted.forward_base(flat_robot)
        else:
            self._set_bn_training(
                self.training
                and self.adapted_bn_mode == "shared_stream_stats"
            )
            human_reference = self.adapted.forward_base(flat_human)
            robot_reference = self.adapted.forward_base(flat_robot)
            robot_adapted_base = self.adapted.forward_base(flat_robot)
        return human_reference, robot_reference, robot_adapted_base

    def _task_aware_pool(
        self,
        features: torch.Tensor,
        language_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, channels, _, _ = features.shape
        tokens = features.permute(0, 1, 3, 4, 2).reshape(
            batch_size, -1, channels
        )
        keys = tokens
        query = language_query
        if self.normalize_visual_tokens_for_attention:
            keys = F.normalize(keys, dim=-1)
        if self.normalize_language_query:
            query = F.normalize(query, dim=-1)

        attention_logits = torch.einsum("bnc,bc->bn", keys, query)
        attention = attention_logits.softmax(dim=1)
        pooled = torch.einsum("bnc,bn->bc", tokens, attention)
        if self.normalize_pooled_features:
            pooled = F.normalize(pooled, dim=-1)
        return pooled, attention

    def forward(
        self,
        human_images: torch.Tensor,
        robot_images: torch.Tensor,
        task_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if human_images.shape != robot_images.shape:
            raise ValueError(
                "Human and robot clips must share shape; got "
                f"{human_images.shape} and {robot_images.shape}."
            )
        if human_images.ndim != 5:
            raise ValueError("Clips must use shape [batch, time, 3, H, W].")
        batch_size, num_frames = human_images.shape[:2]
        flat_human = human_images.flatten(0, 1)
        flat_robot = robot_images.flatten(0, 1)

        with torch.no_grad():
            (
                human_reference,
                robot_reference,
                robot_adapted_base,
            ) = self._forward_visual_streams(flat_human, flat_robot)
        robot_adapted = self.adapted.apply_adapters(robot_adapted_base)

        def restore_time(features: torch.Tensor) -> torch.Tensor:
            return features.reshape(batch_size, num_frames, *features.shape[1:])

        human_reference = restore_time(human_reference)
        robot_reference = restore_time(robot_reference)
        robot_adapted = restore_time(robot_adapted)
        language_query = self.lang_linear(task_embeddings)

        human_pooled, human_attention = self._task_aware_pool(
            human_reference, language_query
        )
        robot_frozen_pooled, robot_frozen_attention = self._task_aware_pool(
            robot_reference, language_query
        )
        robot_adapted_pooled, robot_adapted_attention = self._task_aware_pool(
            robot_adapted, language_query
        )

        return {
            "human_frozen": human_pooled,
            "robot_frozen": robot_frozen_pooled,
            "robot_adapted": robot_adapted_pooled,
            "human_attention": human_attention,
            "robot_frozen_attention": robot_frozen_attention,
            "robot_adapted_attention": robot_adapted_attention,
        }

    def official_model_state(self) -> dict[str, torch.Tensor]:
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.adapted.state_dict().items()
        }
        state.update(self.source_language_state)
        state["lang_linear.weight"] = (
            self.lang_linear.weight.detach().cpu().clone()
        )
        state["lang_linear.bias"] = (
            self.lang_linear.bias.detach().cpu().clone()
        )
        return state

    def trainable_state(self) -> dict[str, torch.Tensor]:
        state = {}
        for key, value in self.state_dict().items():
            if (
                key.startswith("adapted.convnet.late_adapter_")
                or key.startswith("lang_linear.")
                or (
                    key.startswith("adapted.")
                    and any(
                        suffix in key
                        for suffix in (
                            "running_mean",
                            "running_var",
                            "num_batches_tracked",
                        )
                    )
                )
            ):
                state[key] = value.detach().cpu().clone()
        return state

    def load_trainable_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        expected = set(self.trainable_state())
        supplied = set(state)
        missing_state = sorted(expected - supplied)
        unexpected_state = sorted(supplied - expected)
        if missing_state or unexpected_state:
            raise RuntimeError(
                "Resume checkpoint trainable state is incompatible: "
                f"missing={missing_state[:20]}, "
                f"unexpected={unexpected_state[:20]}."
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        return list(missing), list(unexpected)

    def trainable_parameter_count(self) -> dict[str, int]:
        adapter = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if "late_adapter_" in name and parameter.requires_grad
        )
        language_projection = sum(
            parameter.numel() for parameter in self.lang_linear.parameters()
        )
        return {
            "adapters": adapter,
            "language_projection": language_projection,
            "total": adapter + language_projection,
        }
