import torch

from hralign.models import ReleasedLateAdapter


def test_release_adapter_parameter_count():
    adapter = ReleasedLateAdapter()
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 527360


def test_release_zero_mapping_starts_as_identity():
    torch.manual_seed(0)
    adapter = ReleasedLateAdapter()
    features = torch.randn(2, 2048, 2, 2, requires_grad=True)
    output = adapter(features)
    torch.testing.assert_close(output, features)


def test_release_initialization_gradient_signature_is_documented():
    adapter = ReleasedLateAdapter()
    features = torch.randn(1, 2048, 1, 1)
    adapter(features).sum().backward()

    assert adapter.D_fc2.bias.grad is not None
    assert torch.count_nonzero(adapter.D_fc2.bias.grad) > 0
    assert adapter.D_mapping.weight.grad is not None
    assert torch.count_nonzero(adapter.D_mapping.weight.grad) == 0
    assert adapter.D_fc1.weight.grad is not None
    assert torch.count_nonzero(adapter.D_fc1.weight.grad) == 0
