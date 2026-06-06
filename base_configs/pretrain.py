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

"""Default pretraining config values."""

import ml_collections


def get_config():
  """Returns default config."""
  config = ml_collections.ConfigDict()

  # ============================================== #
  # General experiment params.
  # ============================================== #
  # The root directory where experiments will be saved.
  config.root_dir = "/tmp/tcc-core/pretrain_runs/"
  # Rng seed. Set this to `none` to disable seeding.
  config.seed = 1
  # cudnn-related parameters that affect reproducibility.
  config.cudnn_deterministic = False
  config.cudnn_benchmark = True
  # Pretraining algorithm to use.
  config.algorithm = "tcc"
  # Number of steps between tensorboard logging.
  config.logging_frequency = 100
  # Number of steps between consecutive checkpoints.
  config.checkpointing_frequency = 200

  # ============================================== #
  # Dataset params.
  # ============================================== #
  config.data = ml_collections.ConfigDict()

  # Absolute path to the dataset root.
  config.data.root = "/home/paichichi/data/RH20T/TCC_RH20T"
  # The mini-batch size. Note this only specifies the number of videos to
  # load frames from in a single batch. The effective batch size is actually
  # larger since we sample multiple frame sequences per video.
  config.data.batch_size = 4
  # Seed for video/pair sampling. If None, falls back to config.seed.
  config.data.video_sampler_seed = None
  # Which action classes to select for creating the pretraining dataset. Leave
  # it empty to load all action classes.
  config.data.pretrain_action_class = ()
  # Restrict the number of videos per class. This is useful for experiments
  # that test sample complexity based on the number of pretraining
  # demonstrations.
  config.data.max_vids_per_class = -1
  # This controls how a video batch is created. If set to 'random', videos
  # are sampled randomly from different classes. If set to 'same_class', only
  # videos belonging to the same class folder are sampled within a batch.
  # If set to 'paired', adjacent batch entries are sampled as paired sequences
  # from a metadata CSV: [h0, r0, h1, r1, ...].
  config.data.pretraining_video_sampler = "random"
  # Optional metadata path for paired sampling. If empty, each split expects
  # its own metadata at `config.data.root/{train,valid}/metadata.csv`.
  config.data.paired_metadata_path = ""
  # Batch-level dynamic frame count for paired sampling:
  # T = max(min_frames, floor(batch_min_len * ratio)).
  # Set paired_max_frames > 0 to add an explicit upper bound.
  config.data.paired_frame_sample_ratio = 0.5
  config.data.paired_max_frames = -1
  config.data.paired_min_frames = 16
  config.data.paired_drop_short_pairs = True
  config.data.paired_role_order = ("h", "r")

  # ============================================== #
  # Frame sampling params.
  # ============================================== #
  config.frame_sampler = ml_collections.ConfigDict()

  # A wildcard specifying the file extension for images in each video folder.
  # This will usually be either "*.jpg" or "*.png".
  config.frame_sampler.image_ext = "*.png"
  # This controls the type of sampling we perform on video frames.
  config.frame_sampler.strategy = "uniform"
  # The number of frames to sample per video.
  config.frame_sampler.num_frames_per_sequence = 15
  # The number of context frames to sample per frame. This is useful for
  # models that use 3D convolutions.
  config.frame_sampler.num_context_frames = 1
  # The stride between sampled context frames.
  config.frame_sampler.context_stride = 3

  config.frame_sampler.all_sampler = ml_collections.ConfigDict()
  config.frame_sampler.all_sampler.stride = 1

  config.frame_sampler.strided_sampler = ml_collections.ConfigDict()
  config.frame_sampler.strided_sampler.stride = 3
  config.frame_sampler.strided_sampler.offset = True

  config.frame_sampler.uniform_sampler = ml_collections.ConfigDict()
  config.frame_sampler.uniform_sampler.offset = 0

  # Currently, this frame sampler has no additional kwargs.
  config.frame_sampler.window_sampler = ml_collections.ConfigDict()

  # ============================================== #
  # Data augmentation params.
  # ============================================== #
  config.data_augmentation = ml_collections.ConfigDict()

  # The image resolution to train on.
  config.data_augmentation.image_size = (112, 112)
  # A list of image augmentations to apply to the training dataset. note that
  # the order matters, e.g. normalize should be done last if you decide to
  # turn it on.
  config.data_augmentation.train_transforms = [
      "random_resized_crop",
      "color_jitter",
      "grayscale",
      "gaussian_blur",
      # "normalize",
  ]
  # A list of image augmentations to apply to the evaluation dataset.
  config.data_augmentation.eval_transforms = [
      "global_resize",
      # "normalize",
  ]

  # ============================================== #
  # Validation params.
  # ============================================== #
  config.eval = ml_collections.ConfigDict()

  # How many validation dataloader iterations to run. Set to None to evaluate
  # the entire validation dataloader.
  config.eval.val_iters = 20
  # The number of steps between validation passes.
  config.eval.eval_frequency = 500

  # ============================================== #
  # Model params.
  # ============================================== #
  config.model = ml_collections.ConfigDict()

  config.model.model_type = "resnet18_linear"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  config.model.pretrain_path = ""
  config.model.vit_weights = "imagenet"
  config.model.trainable_scope = "all"

  # ============================================== #
  # Loss params.
  # ============================================== #
  config.loss = ml_collections.ConfigDict()

  ## TCC loss.
  config.loss.tcc = ml_collections.ConfigDict()
  config.loss.tcc.stochastic_matching = False
  # If True, only adjacent batch items are aligned:
  # [a0, b0, a1, b1, ...] -> a_i <-> b_i.
  config.loss.tcc.paired_matching = False
  config.loss.tcc.loss_type = "regression_mse"
  config.loss.tcc.cycle_length = 2
  config.loss.tcc.label_smoothing = 0.1
  config.loss.tcc.softmax_temperature = 0.1
  config.loss.tcc.normalize_indices = True
  config.loss.tcc.variance_lambda = 0.001
  config.loss.tcc.huber_delta = 0.1
  config.loss.tcc.similarity_type = "l2"  # cosine

  # ============================================== #
  # Optimizer params
  # ============================================== #
  config.optim = ml_collections.ConfigDict()

  config.optim.train_max_iters = 4_000
  # L2 regularization.
  config.optim.weight_decay = 1e-4
  # Learning rate.
  config.optim.lr = 1e-5

  # ============================================== #
  # End of config file
  # ============================================== #

  return config
