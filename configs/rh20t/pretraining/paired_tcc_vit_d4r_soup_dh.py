# coding=utf-8
"""RH20T paired TCC config using the D4R-SOUP-DH ViT-B/16 backbone."""

from configs.rh20t.pretraining.paired_tcc_vit import get_config as _get_config


def get_config():
  """Returns the RH20T paired TCC D4R-SOUP-DH config."""
  config = _get_config()
  config.root_dir = "/tmp/tcc-core/rh20t_paired_tcc_vit_d4r_soup_dh"
  config.model.pretrain_path = "/home/paichichi/data/pretrain/D4R_SOUP_1M_DH.pth"
  config.model.vit_weights = "none"
  return config
