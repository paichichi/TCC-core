#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.parallel import DistributedDataParallel

from hralign.distributed import (
    gather_with_grad,
    initialize_distributed,
    shutdown_distributed,
)
from hralign.losses import human_robot_contrastive_loss


class Projection(torch.nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimensions, dimensions, bias=False)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.linear(value) for value in inputs)


def main() -> None:
    context = initialize_distributed("cpu")
    try:
        if context.world_size != 2:
            raise RuntimeError("Run with torchrun --nproc-per-node=2.")

        torch.manual_seed(17)
        dimensions = 7
        global_batch = 6
        initial_weight = torch.randn(dimensions, dimensions)
        human = torch.randn(global_batch, dimensions)
        robot_frozen = torch.randn(global_batch, dimensions)
        robot_adapted = torch.randn(global_batch, dimensions)
        local_slice = slice(
            context.rank * (global_batch // context.world_size),
            (context.rank + 1) * (global_batch // context.world_size),
        )

        distributed_model = Projection(dimensions)
        distributed_model.linear.weight.data.copy_(initial_weight)
        wrapped = DistributedDataParallel(distributed_model)
        local_outputs = wrapped(
            human[local_slice],
            robot_frozen[local_slice],
            robot_adapted[local_slice],
        )
        gathered = tuple(gather_with_grad(value) for value in local_outputs)
        distributed_loss, _ = human_robot_contrastive_loss(
            *gathered,
            temperature=0.3,
        )
        distributed_loss.backward()

        reference_model = Projection(dimensions)
        reference_model.linear.weight.data.copy_(initial_weight)
        reference_outputs = reference_model(
            human,
            robot_frozen,
            robot_adapted,
        )
        reference_loss, _ = human_robot_contrastive_loss(
            *reference_outputs,
            temperature=0.3,
        )
        reference_loss.backward()

        assert distributed_model.linear.weight.grad is not None
        assert reference_model.linear.weight.grad is not None
        torch.testing.assert_close(
            distributed_model.linear.weight.grad,
            reference_model.linear.weight.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            distributed_loss,
            reference_loss,
            rtol=1e-6,
            atol=1e-7,
        )
        if context.is_main:
            maximum_error = (
                distributed_model.linear.weight.grad
                - reference_model.linear.weight.grad
            ).abs().max()
            print(
                "DDP global-negative gradient matches single-process batch; "
                f"max_abs_error={float(maximum_error):.3e}"
            )
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
