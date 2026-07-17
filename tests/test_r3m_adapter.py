from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from xirl.models import (
    HRAlignLateAdapter,
    HRAlignR3MBackbone,
    R3M_LATE_ADAPTER_LAYOUT,
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


def test_late_adapter_uses_released_identity_initialization():
  torch.manual_seed(0)
  adapter = HRAlignLateAdapter()

  assert torch.count_nonzero(adapter.D_mapping.weight) == 0
  assert torch.count_nonzero(adapter.D_mapping.bias) == 0
  assert torch.count_nonzero(adapter.D_fc1.bias) == 0
  assert torch.count_nonzero(adapter.D_fc2.bias) == 0
  assert torch.count_nonzero(adapter.D_fc1.weight) > 0
  assert torch.count_nonzero(adapter.D_fc2.weight) > 0

  features = torch.randn(2, 2048, 2, 2)
  torch.testing.assert_close(adapter(features), features)

  adapter(features).square().mean().backward()
  for name, parameter in adapter.named_parameters():
    assert parameter.grad is not None
    nonzero = torch.count_nonzero(parameter.grad).item()
    if name == "D_fc2.bias":
      assert nonzero > 0
    else:
      assert nonzero == 0


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
