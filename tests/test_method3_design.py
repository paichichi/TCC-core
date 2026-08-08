import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from scripts.convert_tcc_r3m_to_rvt import validate_source_format
from scripts import train_multiview_softdtw as trainer
from xirl.models import (
    R3M_ADAPTER_INIT_TRAINABLE,
    R3M_LATE_ADAPTER_LAYOUT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ToyAdapterBackbone(nn.Module):
  output_dim = 4

  def __init__(self):
    super().__init__()
    self.adapter_bias = nn.Parameter(torch.ones(1, 4, 1, 1))
    self.adapter_call_count = 0

  def forward_base(self, images):
    return images

  def apply_adapters(self, features):
    self.adapter_call_count += 1
    return features + self.adapter_bias

  def pool_features(self, features):
    return features.flatten(1)

  def forward(self, images):
    return self.pool_features(self.apply_adapters(self.forward_base(images)))


def test_robot_only_encoding_bypasses_adapter_for_human():
  backbone = ToyAdapterBackbone()
  images = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1)
  group_indices = torch.tensor([0, 0, 1, 1])

  encoded = trainer.encode_backbone_features(
      backbone,
      images,
      group_indices,
      adapter_domain="robot_only",
      human_group_count=1,
  )

  torch.testing.assert_close(encoded[:2], images[:2].flatten(1))
  torch.testing.assert_close(
      encoded[2:], images[2:].flatten(1) + 1.0)
  encoded[2:].sum().backward()
  assert torch.count_nonzero(backbone.adapter_bias.grad) > 0


def test_single_view_both_robot_augmentations_use_same_adapter(monkeypatch):
  backbone = ToyAdapterBackbone()
  monkeypatch.setattr(
      trainer,
      "build_backbone",
      lambda **unused_kwargs: backbone,
  )
  model = trainer.ViewSetAttentionSoftDTW(
      embedding_size=2,
      fusion_size=8,
      pretrain_path="",
      adapter_domain="robot_only",
      view_mode="single",
      representation_mode="backbone_pooled",
  )

  def fail_attention(*unused_args, **unused_kwargs):
    raise AssertionError("single view must bypass attention pooling")

  monkeypatch.setattr(model, "attention_pool", fail_attention)
  projector_calls = []
  hook = model.projector_aux.register_forward_hook(
      lambda unused_module, inputs, output: projector_calls.append(
          (inputs[0].shape, output.shape)))
  aug_a = torch.randn(4, 4, 1, 1)
  aug_b = torch.randn(4, 4, 1, 1)
  group_indices = torch.arange(4)
  camera_ids = torch.zeros(4, dtype=torch.long)
  z_alignment, z_aux_a, z_aux_b = model(
      aug_a,
      group_indices,
      camera_ids,
      num_groups=4,
      aug_images=aug_b,
      aug_group_indices=group_indices,
      aug_camera_ids=camera_ids,
      aug_num_groups=4,
      uniform_views_per_group=1,
      human_group_count=2,
  )
  hook.remove()

  assert z_alignment.shape == (4, 4)
  assert z_aux_a.shape == (4, 2)
  assert z_aux_b.shape == (4, 2)
  assert backbone.adapter_call_count == 2
  assert projector_calls == [
      (torch.Size([4, 4]), torch.Size([4, 2])),
      (torch.Size([4, 4]), torch.Size([4, 2])),
  ]
  assert not hasattr(model, "projector_aux_h")
  assert not hasattr(model, "projector_aux_r")


def test_drop_one_respects_minimum_view_count():
  model = trainer.ViewSetAttentionSoftDTW.__new__(
      trainer.ViewSetAttentionSoftDTW)
  nn.Module.__init__(model)
  model.train()
  mask = torch.ones(3, 4, dtype=torch.bool)

  unchanged = model.sample_feature_view_mask(
      mask, "drop_one", mask_prob=1.0, min_views=4)
  dropped = model.sample_feature_view_mask(
      mask, "drop_one", mask_prob=1.0, min_views=3)

  assert torch.equal(unchanged, mask)
  assert torch.equal(dropped.sum(dim=1), torch.full((3,), 3))


def test_clip_consistent_augmentation_replays_random_parameters(tmp_path):
  first_path = tmp_path / "first.png"
  second_path = tmp_path / "second.png"
  Image.new("RGB", (4, 4), color="red").save(first_path)
  Image.new("RGB", (4, 4), color="blue").save(second_path)

  def random_transform(unused_image):
    return torch.rand(4)

  first = trainer.load_image(
      tmp_path, first_path.name, random_transform, transform_seed=123)
  second = trainer.load_image(
      tmp_path, second_path.name, random_transform, transform_seed=123)
  different = trainer.load_image(
      tmp_path, second_path.name, random_transform, transform_seed=456)

  torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
  assert not torch.equal(first, different)


@pytest.mark.parametrize(
    "loss_fn,extra_args",
    [
        (trainer.compute_multiview_infonce, (0.1,)),
        (trainer.compute_soft_temporal_infonce, (1, 4, 0.1, 0.2, 1.0)),
    ],
)
def test_stopgrad_teacher_updates_only_student(loss_fn, extra_args):
  teacher = torch.randn(4, 8, requires_grad=True)
  student = torch.randn(4, 8, requires_grad=True)
  teacher_norm = torch.nn.functional.normalize(teacher, dim=-1)
  student_norm = torch.nn.functional.normalize(student, dim=-1)

  loss, _ = loss_fn(
      teacher_norm,
      student_norm,
      *extra_args,
      stopgrad_teacher=True,
  )
  loss.backward()

  assert teacher.grad is None
  assert student.grad is not None
  assert torch.count_nonzero(student.grad) > 0


@pytest.mark.parametrize(
    "loss_fn,extra_args",
    [
        (trainer.compute_multiview_infonce, (0.1,)),
        (trainer.compute_soft_temporal_infonce, (1, 4, 0.1, 0.2, 1.0)),
    ],
)
def test_symmetric_contrastive_updates_both_augmentation_branches(
    loss_fn, extra_args):
  aug_a = torch.randn(4, 8, requires_grad=True)
  aug_b = torch.randn(4, 8, requires_grad=True)

  loss, _ = loss_fn(
      torch.nn.functional.normalize(aug_a, dim=-1),
      torch.nn.functional.normalize(aug_b, dim=-1),
      *extra_args,
      stopgrad_teacher=False,
      global_negatives=True,
  )
  loss.backward()

  assert aug_a.grad is not None
  assert aug_b.grad is not None
  assert torch.count_nonzero(aug_a.grad) > 0
  assert torch.count_nonzero(aug_b.grad) > 0


def test_single_view_configuration_is_explicit_and_safe():
  args = Namespace(
      view_mode="auto",
      num_multi_view=1,
      fusion_mode="attention_pool",
      pixel_aug=True,
      view_mask_mode="none",
      view_mask_min_views=1,
      view_mask_prob=0.0,
      prefetch_batches=0,
      lambda_mv=0.5,
      adapter_domain="robot_only",
      backbone="r3m_late_adapter",
      representation_mode="backbone_pooled",
      aux_teacher_stop_grad=False,
      aux_global_negatives=True,
      episode_sampler="global_unique_uniform",
      global_batch_episode_pairs=200,
  )

  trainer.validate_method_configuration(args)
  assert args.view_mode == "single"


def test_locked_single_and_multi_configs_match_protocol():
  single_nesi = trainer.load_config(
      PROJECT_ROOT / "configs/nesi_method3_resnet_single_8ts1v.yaml")
  single_linux = trainer.load_config(
      PROJECT_ROOT / "configs/linux_method3_resnet_single_8ts1v.yaml")
  multi_nesi = trainer.load_config(
      PROJECT_ROOT / "configs/nesi_method3_resnet_ln_8ts4v.yaml")
  multi_linux = trainer.load_config(
      PROJECT_ROOT / "configs/linux_method3_resnet_multi_8ts4v.yaml")

  for single_config in (single_nesi, single_linux):
    assert single_config["seed"] == 1
    assert single_config["num_timestamps"] == 8
    assert single_config["episode_sampler"] == "global_unique_uniform"
    assert single_config["global_batch_episode_pairs"] == 200
    assert single_config["num_multi_view"] == 1
    assert single_config["aux_teacher_stop_grad"] is False
    assert single_config["aux_global_negatives"] is True
  assert single_linux["max_iters"] == 2_200
  assert single_linux["max_iters"] % single_linux["save_every"] == 0
  assert single_nesi["max_iters"] == 40_000

  for multi_config in (multi_nesi, multi_linux):
    assert multi_config["seed"] == 1
    assert multi_config["num_timestamps"] == 8
    assert multi_config["episode_sampler"] == "method3_task_balanced"
    assert multi_config["num_multi_view"] == 4
    assert "global_batch_episode_pairs" not in multi_config
  assert multi_linux["max_iters"] == 23_000
  assert multi_linux["batch_episode_pairs"] == 24
  assert multi_nesi["max_iters"] == 40_000


def test_run_manifest_records_resolved_contract(tmp_path):
  model = nn.Linear(4, 2)
  model.bias.requires_grad = False
  args = Namespace(
      out_dir=tmp_path,
      run_name="unit-test",
      seed=1,
      num_timestamps=8,
      max_iters=40_000,
      resolved_local_pair_batch=50,
      resolved_global_pair_batch=200,
      resolved_world_size=4,
  )

  manifest_path = trainer.write_run_manifest(
      args,
      model,
      eligible_episode_count=9_581,
      episode_index_sha256="a" * 64,
  )
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))

  assert payload["resolved_args"]["seed"] == 1
  assert payload["resolved_args"]["max_iters"] == 40_000
  assert payload["eligible_episode_count"] == 9_581
  assert payload["episode_index_sha256"] == "a" * 64
  assert payload["trainable_parameter_count"] == model.weight.numel()
  assert payload["trainable_parameters"] == [
      {"name": "weight", "numel": model.weight.numel()}
  ]
  assert "commit" in payload["git"]


def test_transfer_rejects_legacy_and_accepts_supported_method3_metadata():
  with pytest.raises(ValueError, match="legacy or incompatible"):
    validate_source_format({"model_format": {}}, require_method3_v2=True)

  model_format = {
      "r3m_late_adapter_layout": R3M_LATE_ADAPTER_LAYOUT,
      "r3m_adapter_init": R3M_ADAPTER_INIT_TRAINABLE,
      "method3_format": "asymmetric_domains_v2",
      "adapter_domain": "robot_only",
      "representation_mode": "backbone_pooled",
  }
  assert validate_source_format(
      {"model_format": model_format},
      require_method3_v2=True,
  ) == model_format

  four_branch_format = {
      "r3m_late_adapter_layout": R3M_LATE_ADAPTER_LAYOUT,
      "r3m_adapter_init": R3M_ADAPTER_INIT_TRAINABLE,
      "method3_format": "shared_adapter_four_branch_v1",
      "adapter_domain": "all_shared_control",
      "representation_mode": "backbone_pooled",
      "method3_objectives": "frozen_teacher_control_v2",
      "control_gain": "frozen_reference_teacher_hinge",
      "spatial_preservation":
          "same_robot_feature_map_relative_residual_hinge",
  }
  assert validate_source_format(
      {"model_format": four_branch_format},
      require_method3_v2=True,
  ) == four_branch_format

  invalid_four_branch_format = dict(four_branch_format)
  invalid_four_branch_format["adapter_domain"] = "robot_only"
  with pytest.raises(ValueError, match="incompatible Method 3 metadata"):
    validate_source_format(
        {"model_format": invalid_four_branch_format},
        require_method3_v2=True,
    )
