# coding=utf-8
"""RH20T paired TCC ResNet18 config with learning rate 1e-4."""

from configs.rh20t.pretraining.paired_tcc_resnet18_imagenet_bs4_f40_lightaug import get_config as _get_config


def get_config():
  """Returns RH20T paired TCC, ResNet18/ImageNet, bs4, 40 frames."""
  config = _get_config()
  config.root_dir = "/tmp/tcc-core/rh20t_paired_tcc_resnet18_imagenet_bs4_f40_lr1e4"
  config.data_augmentation.train_transforms = [
      "random_resized_crop",
      "color_jitter",
      "grayscale",
      "gaussian_blur",
  ]
  config.data_augmentation.eval_transforms = ["global_resize"]
  config.loss.tcc.softmax_temperature = 1.0
  config.optim.lr = 1e-4
  return config
