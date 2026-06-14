# coding=utf-8
"""RH20T paired human-robot TCC config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """Returns the RH20T paired TCC config."""
  config = _get_config()

  config.root_dir = "/tmp/tcc-core/rh20t_paired_tcc"
  config.data.root = "/home/paichichi/data/RH20T/TCC_RH20T"
  config.data.batch_size = 4
  config.data.pretraining_video_sampler = "paired"
  config.data.paired_fixed_frames = 16
  config.data.paired_frame_sample_ratio = 1.0
  config.data.paired_max_frames = 16
  config.data.paired_min_frames = 16
  config.data.paired_drop_short_pairs = True
  config.data.paired_role_order = ("h", "r")
  config.data.paired_distinct_tasks_per_batch = True

  config.frame_sampler.image_ext = "*.jpg"
  config.frame_sampler.strategy = "boundary_stratified"
  config.frame_sampler.use_start_frame = True
  config.frame_sampler.num_frames_per_sequence = 16
  config.frame_sampler.num_context_frames = 1

  config.algorithm = "tcc"
  config.loss.tcc.paired_matching = True
  config.loss.tcc.stochastic_matching = False
  config.loss.tcc.loss_type = "regression_mse"
  config.loss.vvcl.enabled = True
  config.loss.vvcl.weight = 1.0

  return config
