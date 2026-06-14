# coding=utf-8
"""RH20T robot-robot same-camera TCC with ImageNet ResNet18."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """Returns R-R same-task/same-camera ResNet18 TCC config."""
  config = _get_config()

  config.root_dir = "/tmp/tcc-core/rh20t_rr_same_camera_tcc_resnet18_imagenet_bs4_f40"
  config.data.root = "/home/paichichi/data/RH20T/TCC_RH20T"
  config.data.batch_size = 4
  config.data.video_sampler_seed = 1
  config.data.pretraining_video_sampler = "task_role_camera"
  config.data.paired_fixed_frames = 40
  config.data.paired_frame_sample_ratio = 1.0
  config.data.paired_max_frames = 40
  config.data.paired_min_frames = 16
  config.data.paired_drop_short_pairs = True
  config.data.paired_distinct_tasks_per_batch = False
  config.data.task_role_camera_group_keys = ("task_id", "camera_id")
  config.data.task_role_camera_pair_role_order = ("r", "r")
  config.data.task_role_camera_different_camera = False
  config.data.task_role_camera_different_episode = True

  config.frame_sampler.image_ext = "*.jpg"
  config.frame_sampler.strategy = "uniform"
  config.frame_sampler.use_start_frame = True
  config.frame_sampler.num_frames_per_sequence = 40
  config.frame_sampler.num_context_frames = 1

  config.data_augmentation.image_size = (112, 112)
  config.data_augmentation.train_transforms = [
      "random_resized_crop",
      "color_jitter",
      "grayscale",
      "gaussian_blur",
  ]
  config.data_augmentation.eval_transforms = ["global_resize"]

  config.model.model_type = "resnet18_linear"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  config.model.trainable_scope = "all"

  config.algorithm = "tcc"
  config.loss.tcc.enabled = True
  config.loss.tcc.stochastic_matching = False
  config.loss.tcc.paired_matching = True
  config.loss.tcc.loss_type = "regression_mse"
  config.loss.tcc.softmax_temperature = 1.0
  config.loss.tcc.similarity_type = "l2"
  config.loss.soft_dtw.enabled = False
  config.loss.vvcl.enabled = False

  config.optim.weight_decay = 1e-4
  config.optim.lr = 1e-5

  return config
