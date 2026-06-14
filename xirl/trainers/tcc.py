# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TCC trainer."""

from typing import Dict, List, Union

import torch
from xirl.losses import compute_paired_tcc_loss
from xirl.losses import compute_paired_soft_dtw_loss
from xirl.losses import compute_paired_vvcl_siglip_loss
from xirl.losses import compute_tcc_loss
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class TCCTrainer(Trainer):
  """A trainer for Temporal Cycle Consistency Learning [1].

  References:
    [1]: arxiv.org/abs/1904.07846
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    super().__init__(model, optimizer, device, config)

    self.normalize_embeddings = config.model.normalize_embeddings
    self.tcc_enabled = config.loss.tcc.enabled
    self.stochastic_matching = config.loss.tcc.stochastic_matching
    self.paired_matching = config.loss.tcc.paired_matching
    self.loss_type = config.loss.tcc.loss_type
    self.similarity_type = config.loss.tcc.similarity_type
    self.cycle_length = config.loss.tcc.cycle_length
    self.temperature = config.loss.tcc.softmax_temperature
    self.label_smoothing = config.loss.tcc.label_smoothing
    self.variance_lambda = config.loss.tcc.variance_lambda
    self.huber_delta = config.loss.tcc.huber_delta
    self.normalize_indices = config.loss.tcc.normalize_indices
    self.soft_dtw_enabled = config.loss.soft_dtw.enabled
    self.soft_dtw_weight = config.loss.soft_dtw.weight
    self.soft_dtw_mode = config.loss.soft_dtw.mode
    self.soft_dtw_gamma = config.loss.soft_dtw.gamma
    self.soft_dtw_temperature = config.loss.soft_dtw.temperature
    self.soft_dtw_divergence = config.loss.soft_dtw.divergence
    self.soft_dtw_normalize_time = config.loss.soft_dtw.normalize_time
    self.vvcl_enabled = config.loss.vvcl.enabled
    self.vvcl_weight = config.loss.vvcl.weight
    self._base_model = model.module if hasattr(model, "module") else model

    if self.stochastic_matching and self.paired_matching:
      raise ValueError("Paired TCC only supports deterministic matching.")
    if self.soft_dtw_enabled and not self.paired_matching:
      raise ValueError("Soft-DTW auxiliary loss requires paired_matching=True.")
    if self.vvcl_enabled and not self.paired_matching:
      raise ValueError("VVCL auxiliary loss requires paired_matching=True.")
    if self.vvcl_enabled and self._base_model.video_attention_pool is None:
      raise ValueError("VVCL is enabled but the model has no VVCL head.")
    if not self.tcc_enabled and not self.soft_dtw_enabled and not self.vvcl_enabled:
      raise ValueError("At least one of TCC, Soft-DTW, or VVCL must be enabled.")

  def compute_loss(
      self,
      embs,
      batch,
  ):
    if not self.tcc_enabled:
      return embs.new_tensor(0.0)

    steps = batch["frame_idxs"].to(self._device)
    seq_lens = batch["video_len"].to(self._device)

    # Dynamically determine the number of cycles if using stochastic
    # matching.
    batch_size, num_cc_frames = embs.shape[:2]
    num_cycles = int(batch_size * num_cc_frames)

    if self.paired_matching:
      return compute_paired_tcc_loss(
          embs=embs,
          idxs=steps,
          seq_lens=seq_lens,
          normalize_embeddings=self.normalize_embeddings,
          loss_type=self.loss_type,
          similarity_type=self.similarity_type,
          temperature=self.temperature,
          label_smoothing=self.label_smoothing,
          variance_lambda=self.variance_lambda,
          huber_delta=self.huber_delta,
          normalize_indices=self.normalize_indices,
      )

    return compute_tcc_loss(
        embs=embs,
        idxs=steps,
        seq_lens=seq_lens,
        stochastic_matching=self.stochastic_matching,
        normalize_embeddings=self.normalize_embeddings,
        loss_type=self.loss_type,
        similarity_type=self.similarity_type,
        num_cycles=num_cycles,
        cycle_length=self.cycle_length,
        temperature=self.temperature,
        label_smoothing=self.label_smoothing,
        variance_lambda=self.variance_lambda,
        huber_delta=self.huber_delta,
        normalize_indices=self.normalize_indices,
    )

  def compute_auxiliary_loss(
      self,
      out,
      batch,  # pylint: disable=unused-argument
  ):
    losses = []
    if self.soft_dtw_enabled:
      soft_dtw_loss = compute_paired_soft_dtw_loss(
          out.embs,
          gamma=self.soft_dtw_gamma,
          temperature=self.soft_dtw_temperature,
          mode=self.soft_dtw_mode,
          normalize_dimension=(not self.normalize_embeddings),
          divergence=self.soft_dtw_divergence,
          normalize_time=self.soft_dtw_normalize_time,
      )
      losses.append(self.soft_dtw_weight * soft_dtw_loss)

    if self.vvcl_enabled:
      vvcl_tokens = out.vvcl_embs if out.vvcl_embs is not None else out.embs
      video_embs = self._base_model.pool_video_embeddings(vvcl_tokens)
      vvcl_loss = compute_paired_vvcl_siglip_loss(
          video_embs,
          self._base_model.vvcl_logit_scale,
          self._base_model.vvcl_logit_bias,
      )
      losses.append(self.vvcl_weight * vvcl_loss)

    if not losses:
      return out.embs.new_tensor(0.0)
    return torch.stack(losses).sum()
