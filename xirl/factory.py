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

"""API factory."""

import functools
import os.path as osp

import albumentations as alb
import torch
from xirl import frame_samplers
from xirl import models
from xirl import trainers
from xirl import transforms
from xirl import video_samplers
from xirl.dataset import VideoDataset
from xirl.file_utils import get_subdirs
from xirl.types import SequenceType

# Supported image transformations with default args.
TRANSFORMS = {
    "random_resized_crop":
        functools.partial(
            alb.RandomResizedCrop, scale=(0.8, 1.0), ratio=(0.75, 1.333),
            p=1.0),
    "center_crop":
        functools.partial(alb.CenterCrop, p=1.0),
    "global_resize":
        functools.partial(alb.Resize, p=1.0),
    "grayscale":
        functools.partial(alb.ToGray, p=0.2),
    "vertical_flip":
        functools.partial(alb.VerticalFlip, p=0.5),
    "horizontal_flip":
        functools.partial(alb.HorizontalFlip, p=0.5),
    "gaussian_blur":
        functools.partial(
            alb.GaussianBlur,
            blur_limit=(13, 13),
            sigma_limit=(1.0, 2.0),
            p=0.2,
        ),
    "color_jitter":
        functools.partial(
            alb.ColorJitter,
            brightness=0.4,
            contrast=0.4,
            hue=0.1,
            saturation=0.1,
            p=0.8,
        ),
    "rotate":
        functools.partial(alb.Rotate, limit=(-5, 5), border_mode=0, p=0.5),
    "normalize":
        functools.partial(
            alb.Normalize,
            mean=transforms.PretrainedMeans.IMAGENET.value,
            std=transforms.PretrainedStds.IMAGENET.value,
            p=1.0,
        ),
}
FRAME_SAMPLERS = {
    "all": frame_samplers.AllSampler,
    "strided": frame_samplers.StridedSampler,
    "variable_strided": frame_samplers.VariableStridedSampler,
    "uniform": frame_samplers.UniformSampler,
    "uniform_with_positives": frame_samplers.UniformWithPositivesSampler,
    "last_and_randoms": frame_samplers.LastFrameAndRandomFrames,
    "window": frame_samplers.WindowSampler,
}
VIDEO_SAMPLERS = {
    "random": video_samplers.RandomBatchSampler,
    "paired": video_samplers.PairedBatchSampler,
    "same_class": video_samplers.SameClassBatchSampler,
}
MODELS = {
    "resnet18_linear": models.Resnet18LinearEncoderNet,
    "vit_b16_linear": models.ViTB16LinearEncoderNet,
}
TRAINERS = {
    "tcc": trainers.TCCTrainer,
}


def trainer_from_config(config, model, optimizer, device):
  return TRAINERS[config.algorithm](model, optimizer, device, config)


def model_from_config(config):
  """Create a model from a config."""
  kwargs = {
      "num_ctx_frames": config.frame_sampler.num_context_frames,
      "normalize_embeddings": config.model.normalize_embeddings,
      "learnable_temp": config.model.learnable_temp,
  }
  if config.model.model_type in ["resnet18_linear", "vit_b16_linear"]:
    kwargs["embedding_size"] = config.model.embedding_size
  if config.model.model_type == "vit_b16_linear":
    kwargs["pretrain_path"] = config.model.pretrain_path
    kwargs["vit_weights"] = config.model.vit_weights
    kwargs["trainable_scope"] = config.model.trainable_scope
  return MODELS[config.model.model_type](**kwargs)


def optim_from_config(config, model):
  """Create an optimizer from a config."""
  # TODO(kevin): Add SGD and AdamW support.
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("The model has no trainable parameters.")
  return torch.optim.Adam(
      params,
      lr=config.optim.lr,
      weight_decay=config.optim.weight_decay,
  )


def create_transform(name, *args, **kwargs):
  """Create an image augmentation from its name and args."""
  # pylint: disable=invalid-name
  if "::" in name:
    # e.g., `rotate::{'limit': (-45, 45)}`
    name, __kwargs = name.split("::")
    _kwargs = eval(__kwargs)  # pylint: disable=eval-used
  else:
    _kwargs = {}
  _kwargs.update(kwargs)
  return TRANSFORMS[name](*args, **_kwargs)


def frame_sampler_from_config(config):
  """Create a frame sampler from a config."""
  kwargs = {
      "num_frames": config.frame_sampler.num_frames_per_sequence,
      "num_ctx_frames": config.frame_sampler.num_context_frames,
      "ctx_stride": config.frame_sampler.context_stride,
      "pattern": config.frame_sampler.image_ext,
      "seed": config.seed,
  }

  if config.frame_sampler.strategy == "strided":
    kwargs["stride"] = config.frame_sampler.strided_sampler.stride
    kwargs["offset"] = config.frame_sampler.strided_sampler.offset
  elif config.frame_sampler.strategy == "uniform":
    kwargs["offset"] = config.frame_sampler.uniform_sampler.offset

  return FRAME_SAMPLERS[config.frame_sampler.strategy](**kwargs)


def video_sampler_from_config(
    config,
    dir_tree,
    sequential,
    dataset_path=None,
):
  """Create a video sampler from a config."""
  kwargs = {
      "dir_tree": dir_tree,
      "batch_size": config.data.batch_size,
      "sequential": sequential,
  }
  if config.data.pretraining_video_sampler == "paired":
    metadata_path = config.data.paired_metadata_path
    if not metadata_path:
      metadata_path = osp.join(dataset_path, "metadata.csv")
      if not osp.exists(metadata_path):
        metadata_path = osp.join(osp.dirname(dataset_path), "manifest.csv")
    kwargs.update({
        "metadata_path": metadata_path,
        "sample_ratio": config.data.paired_frame_sample_ratio,
        "max_frames": config.frame_sampler.num_frames_per_sequence,
        "min_frames": config.data.paired_min_frames,
        "drop_short_pairs": config.data.paired_drop_short_pairs,
        "role_order": config.data.paired_role_order,
    })
  return VIDEO_SAMPLERS[config.data.pretraining_video_sampler](**kwargs)


def dataset_from_config(config, split, debug):
  """Create a video dataset from a config."""
  dataset_path = osp.join(config.data.root, split)

  image_size = config.data_augmentation.image_size
  if isinstance(image_size, int):
    image_size = (image_size, image_size)
  image_size = tuple(image_size)

  if debug:
    # The minimum data augmentation we want to keep is resizing when
    # debugging.
    aug_names = [
        name for name in config.data_augmentation.eval_transforms
        if "resize" in name or "crop" in name
    ]
    if not aug_names:
      aug_names = ["global_resize"]
  else:
    if split == "train":
      aug_names = config.data_augmentation.train_transforms
    else:
      aug_names = config.data_augmentation.eval_transforms

  # Create a list of data augmentation callables.
  aug_funcs = []
  for name in aug_names:
    if name == "letterbox_resize":
      height, width = image_size
      pad_value = tuple(int(round(channel * 255)) for channel in
                        transforms.PretrainedMeans.IMAGENET.value)
      aug_funcs.extend([
          alb.LongestMaxSize(max_size=max(height, width), p=1.0),
          alb.PadIfNeeded(
              min_height=height,
              min_width=width,
              border_mode=0,
              value=pad_value,
              p=1.0,
          ),
      ])
      continue
    if "resize" in name or "crop" in name:
      aug_funcs.append(create_transform(name, *image_size))
    else:
      aug_funcs.append(create_transform(name))

  augmentor = transforms.VideoAugmentor({SequenceType.FRAMES: aug_funcs})

  # Restrict action classes if they have been provided. Else, load all
  # from the data directory.
  if config.data.pretrain_action_class:
    action_classes = config.data.pretrain_action_class
  else:
    action_classes = get_subdirs(
        dataset_path,
        basename=True,
        nonempty=True,
        sort_lexicographical=True,
    )

  frame_sampler = frame_sampler_from_config(config)
  dataset = VideoDataset(
      dataset_path,
      frame_sampler,
      seed=config.seed,
      augmentor=augmentor,
      max_vids_per_class=config.data.max_vids_per_class,
  )
  dataset.restrict_subdirs(action_classes)

  return dataset
