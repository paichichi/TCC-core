from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from xirl.models import (
    HRAlignLateAdapter,
    HRAlignR3MBackbone,
    R3M_ADAPTER_INIT_RELEASE,
    R3M_ADAPTER_INIT_TRAINABLE,
    R3M_LATE_ADAPTER_LAYOUT,
    build_backbone,
    validate_r3m_adapter_init,
    validate_r3m_adapter_layout,
)


ADAPTED_R3M = Path("/home/paichichi/data/pretrain/AdaptedR3M.pyth")


def apply_official_adapter(adapter, features):
  residual = F.relu(adapter.D_fc1(features))
  residual = F.relu(adapter.D_mapping(residual))
  residual = adapter.D_fc2(residual)
  return features + residual


def official_visual_forward(model, images):
  convnet = model.convnet
  features = convnet.conv1(images)
  features = convnet.bn1(features)
  features = convnet.relu(features)
  features = convnet.maxpool(features)
  features = convnet.layer1(features)
  features = convnet.layer2(features)
  features = convnet.layer3(features)
  features = convnet.layer4(features)
  features = apply_official_adapter(
      convnet.late_adapter_1, features)
  features = apply_official_adapter(
      convnet.late_adapter_2, features)
  features = apply_official_adapter(
      convnet.late_adapter_3, features)
  return torch.flatten(convnet.avgpool(features), 1)


def test_release_initialization_remains_checkpoint_compatible():
  torch.manual_seed(0)
  adapter = HRAlignLateAdapter(
      channels=16,
      hidden_channels=8,
      groups=4,
      init_mode=R3M_ADAPTER_INIT_RELEASE,
  )

  assert torch.count_nonzero(adapter.D_mapping.weight) == 0
  assert torch.count_nonzero(adapter.D_mapping.bias) == 0
  assert torch.count_nonzero(adapter.D_fc1.bias) == 0
  assert torch.count_nonzero(adapter.D_fc2.bias) == 0
  assert torch.count_nonzero(adapter.D_fc1.weight) > 0
  assert torch.count_nonzero(adapter.D_fc2.weight) > 0

  features = torch.randn(2, 16, 2, 2)
  torch.testing.assert_close(adapter(features), features)

  adapter(features).square().mean().backward()
  for name, parameter in adapter.named_parameters():
    assert parameter.grad is not None
    nonzero = torch.count_nonzero(parameter.grad).item()
    if name == "D_fc2.bias":
      assert nonzero > 0
    else:
      assert nonzero == 0


def test_trainable_identity_initialization_has_a_live_gradient_path():
  torch.manual_seed(0)
  adapter = HRAlignLateAdapter(
      channels=16,
      hidden_channels=8,
      groups=4,
      init_mode=R3M_ADAPTER_INIT_TRAINABLE,
  )
  optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
  features = torch.randn(2, 16, 2, 2)
  target = torch.randn_like(features)

  torch.testing.assert_close(adapter(features), features)
  assert torch.count_nonzero(adapter.D_fc1.weight) > 0
  assert torch.count_nonzero(adapter.D_mapping.weight) > 0
  assert torch.count_nonzero(adapter.D_fc2.weight) == 0

  F.mse_loss(adapter(features), target).backward()
  assert torch.count_nonzero(adapter.D_fc2.weight.grad) > 0
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)

  F.mse_loss(adapter(features), target).backward()
  assert torch.count_nonzero(adapter.D_mapping.weight.grad) > 0
  assert torch.count_nonzero(adapter.D_fc1.weight.grad) > 0


def test_backbone_exposes_adapted_and_unadapted_paths():
  torch.manual_seed(0)
  model = HRAlignR3MBackbone(
      train_norm_affine=False,
      train_adapters=True,
      adapter_init=R3M_ADAPTER_INIT_TRAINABLE,
  ).eval()
  images = torch.randn(1, 3, 64, 64)

  with torch.no_grad():
    base_features = model.forward_base(images)
    unadapted_before = model.forward_unadapted(images)
    decomposed = model.pool_features(model.apply_adapters(base_features))
    adapted_before = model(images)
    model.convnet.late_adapter_1.D_fc2.bias.add_(0.1)
    unadapted_after = model.forward_unadapted(images)
    adapted_after = model(images)

  torch.testing.assert_close(adapted_before, decomposed, rtol=0.0, atol=0.0)
  torch.testing.assert_close(
      unadapted_before, unadapted_after, rtol=0.0, atol=0.0)
  assert not torch.equal(adapted_before, adapted_after)


def test_adapter_layout_only_allows_missing_metadata_when_explicit():
  validate_r3m_adapter_layout({}, allow_unversioned=True)
  with pytest.raises(ValueError, match="unversioned"):
    validate_r3m_adapter_layout({}, allow_unversioned=False)
  for allow_unversioned in (False, True):
    with pytest.raises(ValueError, match="incompatible"):
      validate_r3m_adapter_layout(
          {"r3m_late_adapter_layout": "legacy_interleaved"},
          allow_unversioned=allow_unversioned,
      )


def test_adapter_init_provenance_is_strict_for_method3():
  validate_r3m_adapter_init(
      {},
      expected_init=R3M_ADAPTER_INIT_RELEASE,
      allow_unversioned=True,
  )
  with pytest.raises(ValueError, match="without initialization"):
    validate_r3m_adapter_init(
        {},
        expected_init=R3M_ADAPTER_INIT_TRAINABLE,
        allow_unversioned=False,
    )
  with pytest.raises(ValueError, match="incompatible initialization"):
    validate_r3m_adapter_init(
        {"r3m_adapter_init": R3M_ADAPTER_INIT_RELEASE},
        expected_init=R3M_ADAPTER_INIT_TRAINABLE,
        allow_unversioned=False,
    )


@pytest.mark.parametrize(
    "backbone_name,expected_init",
    [
        ("r3m_late_adapter", R3M_ADAPTER_INIT_TRAINABLE),
        ("adapted_r3m", R3M_ADAPTER_INIT_RELEASE),
    ],
)
def test_backbone_alias_selects_explicit_adapter_initialization(
    backbone_name,
    expected_init,
):
  model = build_backbone(
      backbone_name,
      pretrain_path="",
      train_norm_affine=False,
      train_adapters=False,
  )
  assert model.adapter_init == expected_init
  for adapter in (
      model.convnet.late_adapter_1,
      model.convnet.late_adapter_2,
      model.convnet.late_adapter_3,
  ):
    assert adapter.init_mode == expected_init


@pytest.mark.skipif(
    not ADAPTED_R3M.is_file(),
    reason="Released AdaptedR3M checkpoint is unavailable.",
)
def test_released_checkpoint_matches_official_post_layer4_forward():
  torch.manual_seed(0)
  model = HRAlignR3MBackbone(
      ADAPTED_R3M,
      train_norm_affine=False,
      train_adapters=False,
      adapter_init=R3M_ADAPTER_INIT_RELEASE,
      allow_unversioned_adapter_checkpoint=True,
  ).eval()
  images = torch.randn(1, 3, 64, 64)

  with torch.no_grad():
    actual = model(images)
    expected = official_visual_forward(model, images)

  assert model.adapter_layout == R3M_LATE_ADAPTER_LAYOUT
  adapter_parameters = sum(
      parameter.numel()
      for name, parameter in model.named_parameters()
      if ".late_adapter_" in name
  )
  assert adapter_parameters == 1_582_080
  torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
