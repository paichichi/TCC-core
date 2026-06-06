# coding=utf-8
"""RH20T paired TCC config using the D4R-DH ViT-B/16 backbone."""

from configs.rh20t.pretraining.paired_tcc_vit import get_config as _get_config


def get_config():
  """Returns the RH20T paired TCC D4R-DH config."""
  config = _get_config()
  config.root_dir = "/tmp/tcc-core/rh20t_paired_tcc_vit_d4r_dh"
  config.model.pretrain_path = "/home/paichichi/data/pretrain/D4R_DH_1M.pth"
  config.model.vit_weights = "none"
  return config
