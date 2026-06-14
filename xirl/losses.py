"""Loss utilities for RH20T multi-view training."""

from __future__ import annotations

import torch


def batched_pairwise_l2_sq(
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    normalize_dimension: bool = False,
) -> torch.Tensor:
  """Return batched pairwise squared L2 distances with shape [B, M, N]."""
  distances = torch.cdist(seq_x, seq_y).pow(2)
  if normalize_dimension:
    distances = distances / seq_x.shape[-1]
  return distances


def soft_dtw_from_distance(
    distances: torch.Tensor,
    gamma: float,
    normalize_time: bool = True,
) -> torch.Tensor:
  """Compute Soft-DTW from a batched distance matrix.

  Args:
    distances: Pairwise costs with shape [B, M, N].
    gamma: Soft-min smoothing. Smaller values approach hard DTW.
    normalize_time: Divide each distance by M + N.
  """
  if distances.ndim != 3:
    raise ValueError("Soft-DTW distance tensor must have shape [B, M, N].")
  if gamma <= 0:
    raise ValueError("Soft-DTW gamma must be positive.")

  batch_size, num_x, num_y = distances.shape
  inf = torch.full(
      (batch_size,),
      float("inf"),
      device=distances.device,
      dtype=distances.dtype,
  )
  zero = torch.zeros_like(inf)

  previous = [zero] + [inf for _ in range(num_y)]
  for i in range(1, num_x + 1):
    current = [inf]
    for j in range(1, num_y + 1):
      candidates = torch.stack(
          [previous[j], current[j - 1], previous[j - 1]], dim=0)
      soft_min = -gamma * torch.logsumexp(-candidates / gamma, dim=0)
      current.append(distances[:, i - 1, j - 1] + soft_min)
    previous = current

  values = previous[num_y]
  if normalize_time:
    values = values / float(num_x + num_y)
  return values


def soft_dtw_sequence_distance(
    seq_x: torch.Tensor,
    seq_y: torch.Tensor,
    gamma: float,
    normalize_dimension: bool = False,
    divergence: bool = True,
    normalize_time: bool = True,
) -> torch.Tensor:
  """Compute Soft-DTW or Soft-DTW divergence for sequence batches."""
  if seq_x.ndim == 2:
    seq_x = seq_x.unsqueeze(0)
  if seq_y.ndim == 2:
    seq_y = seq_y.unsqueeze(0)
  if seq_x.ndim != 3 or seq_y.ndim != 3:
    raise ValueError("Soft-DTW sequences must have shape [B, T, D].")
  if seq_x.shape[0] != seq_y.shape[0]:
    raise ValueError("Soft-DTW sequence batches must have the same size.")

  dist_xy = batched_pairwise_l2_sq(seq_x, seq_y, normalize_dimension)
  value_xy = soft_dtw_from_distance(dist_xy, gamma, normalize_time)
  if not divergence:
    return value_xy

  dist_xx = batched_pairwise_l2_sq(seq_x, seq_x, normalize_dimension)
  dist_yy = batched_pairwise_l2_sq(seq_y, seq_y, normalize_dimension)
  value_xx = soft_dtw_from_distance(dist_xx, gamma, normalize_time)
  value_yy = soft_dtw_from_distance(dist_yy, gamma, normalize_time)
  return value_xy - 0.5 * (value_xx + value_yy)
