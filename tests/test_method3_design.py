from argparse import Namespace

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


class ToyAdapterBackbone(nn.Module):
  output_dim = 4

  def __init__(self):
    super().__init__()
    self.adapter_bias = nn.Parameter(torch.ones(1, 4, 1, 1))

  def forward_base(self, images):
    return images

  def apply_adapters(self, features):
    return features + self.adapter_bias

  def pool_features(self, features):
    return features.mean(dim=(-2, -1))

  def forward(self, images):
    return self.pool_features(self.apply_adapters(self.forward_base(images)))


class ToyScaleAdapterBackbone(ToyAdapterBackbone):

  def __init__(self):
    super().__init__()
    del self.adapter_bias
    self.adapter_scale = nn.Parameter(torch.tensor(1.5))

  def apply_adapters(self, features):
    return features * self.adapter_scale


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


def test_frozen_robot_control_and_spatial_constraint_only_train_adapter():
  backbone = ToyAdapterBackbone()
  images = torch.arange(
      1, 1 + 4 * 4 * 2 * 2, dtype=torch.float32).reshape(4, 4, 2, 2)
  group_indices = torch.tensor([0, 0, 1, 1])

  encoding = trainer.encode_backbone_feature_bundle(
      backbone,
      images,
      group_indices,
      adapter_domain="robot_only",
      human_group_count=1,
      include_frozen_control=True,
      spatial_preserve_tolerance=0.0,
  )

  assert encoding.frozen_control is not None
  torch.testing.assert_close(
      encoding.adapted[:2], images[:2].mean(dim=(-2, -1)))
  torch.testing.assert_close(
      encoding.frozen_control[2:], images[2:].mean(dim=(-2, -1)))
  assert encoding.spatial_preservation_loss > 0
  encoding.spatial_preservation_loss.backward()
  assert backbone.adapter_bias.grad is not None
  assert torch.count_nonzero(backbone.adapter_bias.grad) > 0
  assert encoding.frozen_control.grad_fn is None


def test_spatial_constraint_has_identity_tolerance_region():
  backbone = ToyAdapterBackbone()
  with torch.no_grad():
    backbone.adapter_bias.zero_()
  images = torch.randn(4, 4, 2, 2)
  group_indices = torch.tensor([0, 0, 1, 1])

  encoding = trainer.encode_backbone_feature_bundle(
      backbone,
      images,
      group_indices,
      adapter_domain="robot_only",
      human_group_count=1,
      include_frozen_control=True,
      spatial_preserve_tolerance=0.05,
  )

  torch.testing.assert_close(
      encoding.spatial_preservation_loss,
      torch.zeros_like(encoding.spatial_preservation_loss),
      atol=1e-8,
      rtol=0.0,
  )
  assert encoding.spatial_violation_fraction == 0


def test_spatial_constraint_detects_magnitude_change_with_same_cosine():
  backbone = ToyScaleAdapterBackbone()
  images = torch.ones(4, 4, 2, 2)
  group_indices = torch.tensor([0, 0, 1, 1])

  encoding = trainer.encode_backbone_feature_bundle(
      backbone,
      images,
      group_indices,
      adapter_domain="robot_only",
      human_group_count=1,
      include_frozen_control=True,
      spatial_preserve_tolerance=0.2,
  )

  # A cosine-only constraint would be exactly zero for positive rescaling.
  torch.testing.assert_close(
      encoding.spatial_relative_delta,
      torch.tensor(0.5),
      atol=1e-7,
      rtol=0.0,
  )
  assert encoding.spatial_preservation_loss > 0
  encoding.spatial_preservation_loss.backward()
  assert backbone.adapter_scale.grad is not None
  assert backbone.adapter_scale.grad > 0


def test_spatial_constraint_is_stable_at_inactive_feature_locations():
  backbone = ToyAdapterBackbone()
  with torch.no_grad():
    backbone.adapter_bias.fill_(0.1)
  images = torch.ones(4, 4, 2, 2)
  # One inactive Robot feature location previously divided by 1e-6 and
  # produced a catastrophic loss despite all active locations being stable.
  images[2:, :, 0, 0] = 0.0
  group_indices = torch.tensor([0, 0, 1, 1])

  encoding = trainer.encode_backbone_feature_bundle(
      backbone,
      images,
      group_indices,
      adapter_domain="robot_only",
      human_group_count=1,
      include_frozen_control=True,
      spatial_preserve_tolerance=0.2,
  )

  assert torch.isfinite(encoding.spatial_preservation_loss)
  assert encoding.spatial_preservation_loss < 1.0
  encoding.spatial_preservation_loss.backward()
  assert backbone.adapter_bias.grad is not None
  assert torch.isfinite(backbone.adapter_bias.grad).all()


def test_single_view_forward_never_calls_attention(monkeypatch):
  monkeypatch.setattr(
      trainer,
      "build_backbone",
      lambda **unused_kwargs: ToyAdapterBackbone(),
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
  clean = torch.randn(4, 4, 1, 1)
  augmented = torch.randn(4, 4, 1, 1)
  group_indices = torch.arange(4)
  camera_ids = torch.zeros(4, dtype=torch.long)
  z_alignment, z_clean, z_augmented = model(
      clean,
      group_indices,
      camera_ids,
      num_groups=4,
      aug_images=augmented,
      aug_group_indices=group_indices,
      aug_camera_ids=camera_ids,
      aug_num_groups=4,
      uniform_views_per_group=1,
      human_group_count=2,
  )

  assert z_alignment.shape == (4, 4)
  assert z_clean.shape == (4, 2)
  assert z_augmented.shape == (4, 2)


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


def test_control_gain_uses_frozen_counterfactual_without_its_gradients():
  torch.manual_seed(7)
  h_adapted = torch.randn(2, 4, 8, requires_grad=True)
  r_adapted = torch.randn(2, 4, 8, requires_grad=True)
  h_frozen = torch.randn(2, 4, 8, requires_grad=True)
  r_frozen = torch.randn(2, 4, 8, requires_grad=True)

  _, metrics, gain_loss = trainer.compute_controlled_soft_alignment_paired(
      h_adapted,
      r_adapted,
      h_frozen,
      r_frozen,
      temperature=0.1,
      feature_mode="raw",
      normalize_epsilon=1e-12,
      epsilon=0.05,
      rho=0.5,
      teacher_feature_weight=1.0,
      sinkhorn_iters=5,
      struct_lambda=0.1,
      max_forward_step=1.0,
      # A large margin guarantees an active hinge for this gradient test.
      margin=10.0,
  )
  gain_loss.backward()

  assert gain_loss > 0
  assert h_adapted.grad is not None
  assert r_adapted.grad is not None
  assert torch.count_nonzero(h_adapted.grad) > 0
  assert torch.count_nonzero(r_adapted.grad) > 0
  assert h_frozen.grad is None
  assert r_frozen.grad is None
  assert set((
      "control_adapted_score",
      "control_frozen_score",
      "control_gain_gap",
      "control_win_rate",
  )).issubset(metrics)


def test_equal_adapted_and_frozen_control_has_zero_margin_loss():
  torch.manual_seed(11)
  human = torch.randn(2, 4, 8, requires_grad=True)
  robot = torch.randn(2, 4, 8, requires_grad=True)

  _, _, gain_loss = trainer.compute_controlled_soft_alignment_paired(
      human,
      robot,
      human.detach().clone(),
      robot.detach().clone(),
      temperature=0.1,
      feature_mode="raw",
      normalize_epsilon=1e-12,
      epsilon=0.05,
      rho=0.5,
      teacher_feature_weight=1.0,
      sinkhorn_iters=5,
      struct_lambda=0.1,
      max_forward_step=1.0,
      margin=0.0,
  )

  torch.testing.assert_close(
      gain_loss, torch.zeros_like(gain_loss), atol=1e-7, rtol=0.0)


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
      aux_teacher_stop_grad=True,
      aux_global_negatives=True,
  )

  trainer.validate_method_configuration(args)
  assert args.view_mode == "single"


def test_vit_single_stays_on_two_branch_non_control_path():
  args = Namespace(
      view_mode="single",
      num_multi_view=1,
      fusion_mode="attention_pool",
      pixel_aug=True,
      view_mask_mode="none",
      view_mask_min_views=1,
      view_mask_prob=0.0,
      prefetch_batches=0,
      lambda_mv=0.5,
      lambda_control_gain=0.0,
      lambda_spatial_preserve=0.0,
      adapter_domain="all",
      backbone="vit_b16",
      representation_mode="legacy_projected",
      aux_teacher_stop_grad=False,
      aux_global_negatives=False,
  )

  trainer.validate_method_configuration(args)

  args.adapter_domain = "all_shared_control"
  with pytest.raises(ValueError, match="R3M adapter backbone"):
    trainer.validate_method_configuration(args)


def test_control_objectives_reject_non_asymmetric_configuration():
  args = Namespace(
      view_mode="multi",
      num_multi_view=4,
      fusion_mode="attention_pool",
      pixel_aug=False,
      view_mask_mode="drop_one",
      view_mask_min_views=3,
      view_mask_prob=1.0,
      prefetch_batches=0,
      lambda_mv=0.5,
      lambda_control_gain=1.0,
      control_gain_margin=0.0,
      lambda_spatial_preserve=0.1,
      spatial_preserve_tolerance=0.05,
      adapter_domain="all",
      backbone="r3m_late_adapter",
      train_backbone_adapters=True,
      representation_mode="backbone_pooled",
      softdtw_mode="soft_alignment",
      aux_teacher_stop_grad=True,
      aux_global_negatives=True,
  )

  with pytest.raises(ValueError, match="controlled adapter domain"):
    trainer.validate_method_configuration(args)


def test_transfer_rejects_legacy_and_accepts_asymmetric_v2_metadata():
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


def test_transfer_accepts_counterfactual_spatial_objective_metadata():
  model_format = {
      "r3m_late_adapter_layout": R3M_LATE_ADAPTER_LAYOUT,
      "r3m_adapter_init": R3M_ADAPTER_INIT_TRAINABLE,
      "method3_format": "asymmetric_domains_v2",
      "method3_objectives": "counterfactual_spatial_v1",
      "adapter_domain": "robot_only",
      "representation_mode": "backbone_pooled",
      "control_gain": "shared_teacher_hinge",
      "spatial_preservation":
          "same_robot_feature_map_relative_residual_hinge",
  }
  assert validate_source_format(
      {"model_format": model_format},
      require_method3_v2=True,
  ) == model_format

  incomplete = dict(model_format)
  del incomplete["spatial_preservation"]
  with pytest.raises(ValueError, match="objective metadata"):
    validate_source_format(
        {"model_format": incomplete},
        require_method3_v2=True,
    )
