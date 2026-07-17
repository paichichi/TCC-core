import torch

from hralign.distributed import initialize_distributed


def test_bare_cuda_device_resolves_to_local_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    context = initialize_distributed("cuda")

    assert context.device == torch.device("cuda:0")
    assert selected == [torch.device("cuda:0")]
