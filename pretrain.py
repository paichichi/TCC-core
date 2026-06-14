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

"""Launch script for pre-training representations."""

import csv
import os
import os.path as osp

from absl import app
from absl import flags
from absl import logging
from base_configs import validate_config
from ml_collections import config_flags
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from torchkit.utils.py_utils import Stopwatch
from utils import load_config_from_dir
from utils import setup_experiment
from utils import update_config_from_yaml
from xirl import common
from xirl import factory

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_boolean("resume", False, "Whether to resume training.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")
flags.DEFINE_string(
    "gpus",
    None,
    "Comma-separated GPU ids to expose, e.g. '0' or '0,1,2,3'. For multiple "
    "GPUs, launch with torchrun and set --nproc_per_node to the GPU count.",
)
flags.DEFINE_boolean("raw_imagenet", False, "")
flags.DEFINE_string(
    "config_yaml",
    None,
    "Optional YAML file whose values override the Python config.",
)
flags.DEFINE_integer(
    "max_iters",
    None,
    "Optional override for config.optim.train_max_iters.",
)
flags.DEFINE_integer(
    "log_every",
    None,
    "Optional override for config.logging_frequency.",
)
flags.DEFINE_integer(
    "eval_every",
    None,
    "Optional override for config.eval.eval_frequency.",
)
flags.DEFINE_integer(
    "checkpoint_every",
    None,
    "Optional override for config.checkpointing_frequency.",
)

config_flags.DEFINE_config_file(
    "config",
    "base_configs/pretrain.py",
    "File path to the training hyperparameter configuration.",
)


def _configure_visible_gpus():
  """Restrict visible CUDA devices before CUDA is initialized."""
  if not FLAGS.gpus:
    return
  gpus = FLAGS.gpus.replace("gpu", "").replace("GPU", "").replace(" ", "")
  if not gpus:
    raise ValueError("--gpus was provided but no GPU id was found.")
  for gpu_id in gpus.split(","):
    if not gpu_id.isdigit():
      raise ValueError(f"Invalid GPU id in --gpus={FLAGS.gpus}")
  os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def _init_distributed():
  """Initialize DDP when launched by torchrun."""
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  distributed = world_size > 1

  if distributed:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
      torch.cuda.set_device(local_rank)

  return distributed, rank, local_rank, world_size


def _resolve_device(distributed, local_rank):
  if torch.cuda.is_available():
    if distributed:
      return torch.device("cuda", local_rank)
    return torch.device(FLAGS.device)
  logging.info("No GPU device found. Falling back to CPU.")
  return torch.device("cpu")


def _reduce_loss_dict(loss_dict, distributed, world_size):
  """Average scalar loss tensors across ranks for logging."""
  if not distributed:
    return loss_dict

  reduced = {}
  for key, value in loss_dict.items():
    if torch.is_tensor(value):
      tensor = value.detach().clone()
      dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
      tensor /= world_size
      reduced[key] = tensor
    else:
      reduced[key] = value
  return reduced


def _open_loss_csv(exp_dir, resume):
  path = osp.join(exp_dir, "losses.csv")
  exists = osp.exists(path)
  fp = open(path, "a" if resume else "w", newline="")
  writer = csv.writer(fp)
  if not resume or not exists:
    writer.writerow([
        "global_step",
        "epoch",
        "seconds_per_iter",
        "train_base_loss",
        "train_auxiliary_loss",
        "train_total_loss",
    ])
    fp.flush()
  return fp, writer


def _write_loss_csv(writer, fp, global_step, epoch, seconds_per_iter, loss_dict):
  def _to_float(value):
    if torch.is_tensor(value):
      return value.item()
    return float(value)

  writer.writerow([
      global_step,
      epoch,
      f"{seconds_per_iter:.6f}",
      f"{_to_float(loss_dict['train/base_loss']):.8f}",
      f"{_to_float(loss_dict['train/auxiliary_loss']):.8f}",
      f"{_to_float(loss_dict['train/total_loss']):.8f}",
  ])
  fp.flush()


@experiment.pdb_fallback
def main(_):
  _configure_visible_gpus()
  distributed, rank, local_rank, world_size = _init_distributed()
  is_main = rank == 0

  if FLAGS.config_yaml:
    update_config_from_yaml(FLAGS.config, FLAGS.config_yaml)
  if FLAGS.max_iters is not None:
    FLAGS.config.optim.train_max_iters = FLAGS.max_iters
  if FLAGS.log_every is not None:
    FLAGS.config.logging_frequency = FLAGS.log_every
  if FLAGS.eval_every is not None:
    FLAGS.config.eval.eval_frequency = FLAGS.eval_every
  if FLAGS.checkpoint_every is not None:
    FLAGS.config.checkpointing_frequency = FLAGS.checkpoint_every

  # Make sure we have a valid config that inherits all the keys defined in the
  # base config.
  validate_config(FLAGS.config, mode="pretrain")

  config = FLAGS.config
  exp_dir = osp.join(config.root_dir, FLAGS.experiment_name)
  if is_main:
    setup_experiment(exp_dir, config, FLAGS.resume)
  if distributed:
    dist.barrier()
  if distributed and not is_main and FLAGS.resume:
    load_config_from_dir(exp_dir, config)

  # No need to do any pretraining if we're loading the raw pretrained
  # ImageNet baseline.
  if FLAGS.raw_imagenet:
    return

  device = _resolve_device(distributed, local_rank)
  logging.info(
      "Using device: %s (rank=%d, local_rank=%d, world_size=%d)",
      device,
      rank,
      local_rank,
      world_size,
  )

  # Set RNG seeds.
  if config.seed is not None:
    logging.info("Pretraining experiment seed: %d", config.seed)
    experiment.seed_rngs(config.seed)
    experiment.set_cudnn(config.cudnn_deterministic, config.cudnn_benchmark)
  else:
    logging.info("No RNG seed has been set for this pretraining experiment.")

  logger = Logger(osp.join(exp_dir, "tb"), FLAGS.resume) if is_main else None
  loss_csv_fp = None
  loss_csv_writer = None
  if is_main:
    loss_csv_fp, loss_csv_writer = _open_loss_csv(exp_dir, FLAGS.resume)

  # Build model and optimizer before changing sampler seeds so all ranks start
  # from identical weights while still seeing different shuffled video batches.
  model = factory.model_from_config(config)
  optimizer = factory.optim_from_config(config, model)
  checkpoint_model = model

  if distributed:
    base_sampler_seed = (
        config.seed if config.data.video_sampler_seed is None
        else config.data.video_sampler_seed)
    if base_sampler_seed is not None:
      config.data.video_sampler_seed = base_sampler_seed + rank

  pretrain_loaders = common.get_pretraining_dataloaders(config)
  model.to(device)
  train_model = model
  if distributed:
    train_model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        broadcast_buffers=False,
    )
  trainer = factory.trainer_from_config(config, train_model, optimizer, device)

  # Create checkpoint manager.
  checkpoint_dir = osp.join(exp_dir, "checkpoints")
  checkpoint_manager = CheckpointManager(
      checkpoint_dir,
      model=checkpoint_model,
      optimizer=optimizer,
  )

  global_step = checkpoint_manager.restore_or_initialize()
  if distributed:
    dist.barrier()
  total_batches = max(1, len(pretrain_loaders["train"]))
  epoch = int(global_step / total_batches)
  complete = False
  stopwatch = Stopwatch()
  try:
    while not complete:
      for batch in pretrain_loaders["train"]:
        train_loss = trainer.train_one_iter(batch)
        train_loss = _reduce_loss_dict(train_loss, distributed, world_size)

        time_per_iter = stopwatch.elapsed()
        if is_main and loss_csv_writer is not None:
          _write_loss_csv(
              loss_csv_writer,
              loss_csv_fp,
              global_step,
              epoch,
              time_per_iter,
              train_loss,
          )

        if is_main and not global_step % config.logging_frequency:
          for k, v in train_loss.items():
            logger.log_scalar(v, global_step, k, "pretrain")
          logger.flush()

        if config.eval.eval_frequency > 0 and not global_step % config.eval.eval_frequency:
          # Evaluate the model on the pretraining validation dataset.
          valid_loss = trainer.eval_num_iters(
              pretrain_loaders["valid"],
              config.eval.val_iters,
          )
          valid_loss = _reduce_loss_dict(valid_loss, distributed, world_size)
          if is_main:
            for k, v in valid_loss.items():
              logger.log_scalar(v, global_step, k, "pretrain")

        # Save model checkpoint.
        if (is_main and config.checkpointing_frequency > 0 and
            not global_step % config.checkpointing_frequency):
          checkpoint_manager.save(global_step)

        # Exit if complete.
        global_step += 1
        if global_step >= config.optim.train_max_iters:
          complete = True
          break

        if is_main:
          logging.info(
              "Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}".format(
                  global_step,
                  config.optim.train_max_iters,
                  epoch,
                  time_per_iter,
                  train_loss["train/total_loss"].item(),
              ))
        stopwatch.reset()
      epoch += 1

  except KeyboardInterrupt:
    logging.info("Caught keyboard interrupt. Saving model before quitting.")

  finally:
    if is_main:
      checkpoint_manager.save(global_step)
      logger.close()
      if loss_csv_fp is not None:
        loss_csv_fp.close()
    if distributed:
      dist.barrier()
      dist.destroy_process_group()


if __name__ == "__main__":
  flags.mark_flag_as_required("experiment_name")
  app.run(main)
