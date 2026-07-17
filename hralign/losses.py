from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def human_robot_contrastive_loss(
    human_frozen: torch.Tensor,
    robot_frozen: torch.Tensor,
    robot_adapted: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Implements Eq. (6) of HR-Align.

    Each direction uses all adapted cross-pair features as candidates and adds
    the paired unadapted robot similarity as one extra baseline candidate.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not (
        human_frozen.shape == robot_frozen.shape == robot_adapted.shape
    ):
        raise ValueError(
            "human_frozen, robot_frozen, and robot_adapted must share shape; "
            f"got {human_frozen.shape}, {robot_frozen.shape}, "
            f"{robot_adapted.shape}."
        )
    if human_frozen.ndim != 2:
        raise ValueError("Expected feature tensors with shape [batch, channels].")

    batch_size = human_frozen.shape[0]
    if batch_size == 0:
        raise ValueError("The contrastive batch cannot be empty.")

    targets = torch.arange(batch_size, device=human_frozen.device)
    baseline = (human_frozen * robot_frozen).sum(dim=-1, keepdim=True)
    baseline = baseline / temperature

    human_to_robot = human_frozen @ robot_adapted.transpose(0, 1)
    human_to_robot = human_to_robot / temperature
    h2r_candidates = torch.cat([human_to_robot, baseline], dim=1)
    loss_h2r = F.cross_entropy(h2r_candidates, targets)

    robot_to_human = robot_adapted @ human_frozen.transpose(0, 1)
    robot_to_human = robot_to_human / temperature
    r2h_candidates = torch.cat([robot_to_human, baseline], dim=1)
    loss_r2h = F.cross_entropy(r2h_candidates, targets)

    loss = 0.5 * (loss_h2r + loss_r2h)
    diagonal = human_to_robot.diagonal()
    if batch_size > 1:
        off_diagonal = human_to_robot[
            ~torch.eye(
                batch_size, dtype=torch.bool, device=human_to_robot.device
            )
        ]
        off_mean = off_diagonal.mean()
    else:
        off_mean = human_to_robot.new_zeros(())

    metrics = {
        "loss_h2r": loss_h2r.detach(),
        "loss_r2h": loss_r2h.detach(),
        "positive_similarity": (diagonal * temperature).mean().detach(),
        "frozen_similarity": (baseline.squeeze(1) * temperature).mean().detach(),
        "negative_similarity": (off_mean * temperature).detach(),
        "h2r_top1": (
            human_to_robot.argmax(dim=1) == targets
        ).float().mean().detach(),
        "r2h_top1": (
            robot_to_human.argmax(dim=1) == targets
        ).float().mean().detach(),
    }
    return loss, metrics


def scalar_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}
