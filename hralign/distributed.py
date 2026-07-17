from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    requested_device: str | None = None,
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if requested_device:
        requested = torch.device(requested_device)
        if requested.type == "cuda" and requested.index is None:
            device = torch.device("cuda", local_rank)
        elif world_size > 1 and requested.type == "cuda":
            if requested.index != local_rank:
                raise ValueError(
                    "Under torchrun, an explicit CUDA device must match "
                    f"LOCAL_RANK={local_rank}; got {requested}."
                )
            device = requested
        else:
            device = requested
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=30),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(list(all_gather(tensor)), dim=0)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
