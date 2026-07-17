from __future__ import annotations

import argparse
import csv
from itertools import islice
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import (
    export_official_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .config import dump_config, load_config
from .data import RH20TPairDataset, sampling_config_from_dict, seed_worker
from .distributed import (
    DistributedContext,
    barrier,
    gather_with_grad,
    initialize_distributed,
    reduce_mean,
    shutdown_distributed,
)
from .losses import human_robot_contrastive_loss
from .models import HRAlignR3ML
from .text import FrozenR3MTextEncoder, PrecomputedTaskTextEncoder


CSV_FIELDS = [
    "step",
    "epoch",
    "loss",
    "loss_h2r",
    "loss_r2h",
    "positive_similarity",
    "frozen_similarity",
    "negative_similarity",
    "h2r_top1",
    "r2h_top1",
    "adapted_embedding_std",
    "adapted_attention_entropy",
    "learning_rate",
    "grad_norm",
    "seconds",
    "images",
    "cuda_memory_mb",
]


class EpochDistributedSampler(DistributedSampler):
    """Distributed sampler with resumable offsets and epoch-aware indices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        super().set_epoch(epoch)
        if not 0 <= start_index <= super().__len__():
            raise ValueError(
                f"Sampler start_index={start_index} is outside "
                f"[0, {super().__len__()}]."
            )
        self.start_index = int(start_index)

    def __iter__(self):
        indices = islice(super().__iter__(), self.start_index, None)
        return iter((index, self.epoch) for index in indices)

    def __len__(self) -> int:
        return super().__len__() - self.start_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the release-compatible HR-Align R3M-Align-L model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a YAML value; may be repeated.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def seed_everything(seed: int, rank: int) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    zero_weight_decay_1d: bool,
) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if zero_weight_decay_1d and (
            parameter.ndim <= 1 or name.endswith(".bias")
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {
            "params": decay,
            "weight_decay": weight_decay,
            "layer_decay": 1.0,
            "apply_LARS": False,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
            "layer_decay": 1.0,
            "apply_LARS": False,
        },
    ]
    return torch.optim.Adam(groups, lr=learning_rate, betas=(0.9, 0.999))


class SlowFastEpochLRScheduler:
    """Step-addressable transcription of SlowFast's epoch-based cosine LR."""

    format_version = 1

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        schedule_epochs: float,
        steps_per_epoch: int,
        end_lr: float,
    ):
        if base_lr <= 0 or warmup_start_lr < 0 or end_lr < 0:
            raise ValueError(
                "Learning rates must be non-negative and base_lr positive."
            )
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if schedule_epochs <= warmup_epochs:
            raise ValueError("schedule_epochs must be greater than warmup_epochs.")
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive.")
        if end_lr >= base_lr:
            raise ValueError("end_lr must be smaller than base_lr.")

        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_start_lr = float(warmup_start_lr)
        self.warmup_epochs = float(warmup_epochs)
        self.schedule_epochs = float(schedule_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.end_lr = float(end_lr)
        self.last_completed_step = 0
        self.set_step(0)

    def lr_at_step(self, completed_steps: int) -> float:
        if completed_steps < 0:
            raise ValueError("completed_steps cannot be negative.")
        epoch_exact = completed_steps / self.steps_per_epoch
        if self.warmup_epochs > 0 and epoch_exact < self.warmup_epochs:
            progress = epoch_exact / self.warmup_epochs
            return self.warmup_start_lr + progress * (
                self.base_lr - self.warmup_start_lr
            )

        progress = (
            epoch_exact - self.warmup_epochs
        ) / (self.schedule_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end_lr + (self.base_lr - self.end_lr) * cosine

    def set_step(self, completed_steps: int) -> float:
        learning_rate = self.lr_at_step(completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("layer_decay", 1.0))
        self.last_completed_step = int(completed_steps)
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "last_completed_step": self.last_completed_step,
            "base_lr": self.base_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "warmup_epochs": self.warmup_epochs,
            "schedule_epochs": self.schedule_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "end_lr": self.end_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != self.format_version:
            raise ValueError("Unsupported HR-Align scheduler checkpoint format.")
        expected = {
            "base_lr": self.base_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "warmup_epochs": self.warmup_epochs,
            "schedule_epochs": self.schedule_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "end_lr": self.end_lr,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Scheduler configuration changed across resume: "
                f"{mismatches}"
            )
        self.set_step(int(state["last_completed_step"]))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    warmup_start_lr: float,
    warmup_epochs: float,
    schedule_epochs: float,
    steps_per_epoch: int,
    end_lr: float,
) -> SlowFastEpochLRScheduler:
    return SlowFastEpochLRScheduler(
        optimizer=optimizer,
        base_lr=base_lr,
        warmup_start_lr=warmup_start_lr,
        warmup_epochs=warmup_epochs,
        schedule_epochs=schedule_epochs,
        steps_per_epoch=steps_per_epoch,
        end_lr=end_lr,
    )


def build_text_encoder(
    config: dict[str, Any],
    model: HRAlignR3ML,
    device: torch.device,
) -> torch.nn.Module:
    text_config = config.get("text", {})
    embeddings_path = text_config.get("embeddings")
    if embeddings_path:
        encoder = PrecomputedTaskTextEncoder(embeddings_path)
    else:
        encoder = FrozenR3MTextEncoder(
            model.source_language_state,
            tokenizer_name=text_config.get(
                "tokenizer_name", "distilbert-base-uncased"
            ),
            cache_dir=text_config.get("cache_dir") or None,
            local_files_only=bool(text_config.get("local_files_only", False)),
            cache_by_text=bool(text_config.get("cache_by_text", False)),
        )
    encoder.to(device)
    encoder.eval()
    return encoder


def encode_text_batch(
    encoder: torch.nn.Module,
    batch: dict[str, Any],
) -> torch.Tensor:
    if isinstance(encoder, PrecomputedTaskTextEncoder):
        return encoder(batch["task_id"], batch["task_text"])
    return encoder(batch["task_text"])


def _mean_metric(value: torch.Tensor) -> float:
    return float(reduce_mean(value).cpu())


def _open_csv(path: Path, resume: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    has_content = resume and path.exists() and path.stat().st_size > 0
    handle = path.open("a" if resume else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    if not has_content:
        writer.writeheader()
        handle.flush()
    return handle, writer


def _memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def train(config: dict[str, Any], args: argparse.Namespace) -> None:
    context = initialize_distributed(args.device)
    try:
        _train(config, args, context)
    finally:
        shutdown_distributed()


def _train(
    config: dict[str, Any],
    args: argparse.Namespace,
    context: DistributedContext,
) -> None:
    train_config = config.get("train", {})
    seed = int(train_config.get("seed", 0))
    seed_everything(seed, context.rank)

    output_dir = Path(train_config["output_dir"])
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, output_dir / "config_resolved.yaml")
    barrier()

    data_config = config["data"]
    sampling = sampling_config_from_dict(config.get("sampling", {}))
    dataset = RH20TPairDataset(
        data_root=data_config["root"],
        lookup_path=data_config["lookup"],
        manifest_path=data_config["manifest"],
        task_descriptions_path=data_config["task_descriptions"],
        sampling=sampling,
        max_pairs=data_config.get("max_pairs"),
        subset_seed=int(data_config.get("subset_seed", 0)),
        allowed_tasks=data_config.get("allowed_tasks"),
        train=True,
        augmentation_seed=seed,
    )
    sampler = EpochDistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    batch_size = int(train_config.get("batch_size_per_gpu", 50))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=(
            context.device.type == "cuda"
            and bool(train_config.get("pin_memory", True))
        ),
        persistent_workers=(
            int(train_config.get("num_workers", 4)) > 0
            and bool(train_config.get("persistent_workers", True))
        ),
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    if not len(loader):
        raise RuntimeError("DataLoader has no complete batches.")
    steps_per_epoch = len(loader)

    model_config = config.get("model", {})
    core_model = HRAlignR3ML(
        pretrain_path=model_config["pretrain"],
        adapted_bn_mode=model_config.get(
            "adapted_bn_mode", "shared_stream_stats"
        ),
        normalize_language_query=bool(
            model_config.get("normalize_language_query", False)
        ),
        normalize_visual_tokens_for_attention=bool(
            model_config.get(
                "normalize_visual_tokens_for_attention", False
            )
        ),
        normalize_pooled_features=bool(
            model_config.get("normalize_pooled_features", True)
        ),
    ).to(context.device)
    text_encoder = build_text_encoder(config, core_model, context.device)

    learning_rate = float(train_config.get("learning_rate", 1e-4))
    optimizer = build_optimizer(
        core_model,
        learning_rate=learning_rate,
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
        zero_weight_decay_1d=bool(
            train_config.get("zero_weight_decay_1d", True)
        ),
    )
    scheduler = build_scheduler(
        optimizer,
        base_lr=learning_rate,
        warmup_start_lr=float(train_config.get("warmup_start_lr", 1e-6)),
        warmup_epochs=float(train_config.get("warmup_epochs", 10.0)),
        schedule_epochs=float(train_config.get("schedule_epochs", 300.0)),
        steps_per_epoch=steps_per_epoch,
        end_lr=float(train_config.get("end_lr", 1e-6)),
    )
    amp_enabled = bool(train_config.get("amp", False))
    scaler = torch.amp.GradScaler(
        context.device.type,
        enabled=amp_enabled and context.device.type == "cuda",
    )

    model: torch.nn.Module = core_model
    if context.world_size > 1:
        model = DistributedDataParallel(
            core_model,
            device_ids=(
                [context.device.index]
                if context.device.type == "cuda"
                else None
            ),
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

    start_step = 0
    saved_epoch = 0
    resume_path = args.resume or train_config.get("resume")
    if resume_path:
        start_step, saved_epoch = load_training_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    actual_global_batch = batch_size * context.world_size
    expected_global_batch = int(
        train_config.get("expected_global_batch_size", 200)
    )
    if context.is_main:
        counts = core_model.trainable_parameter_count()
        task_count = len({record.task_id for record in dataset.records})
        print(
            f"pairs={len(dataset)} tasks={task_count} "
            f"frames={sampling.num_frames} batch_per_gpu={batch_size} "
            f"world_size={context.world_size} "
            f"contrastive_batch={actual_global_batch} "
            f"steps_per_epoch={steps_per_epoch}"
        )
        print(
            f"trainable adapters={counts['adapters']} "
            f"language_projection={counts['language_projection']} "
            f"total={counts['total']}"
        )
        if actual_global_batch != expected_global_batch:
            print(
                "WARNING: the paper uses contrastive batch 200; this run uses "
                f"{actual_global_batch}. Gradient accumulation cannot recreate "
                "the missing in-batch negatives."
            )

    csv_handle = None
    csv_writer = None
    if context.is_main:
        csv_handle, csv_writer = _open_csv(
            output_dir / "losses.csv", resume=bool(resume_path)
        )

    max_steps = int(train_config.get("max_steps", 8000))
    log_every = int(train_config.get("log_every", 10))
    save_every = int(train_config.get("save_every", 1000))
    temperature = float(config.get("loss", {}).get("temperature", 0.1))
    grad_clip = float(train_config.get("grad_clip_norm", 1.0))
    epoch = start_step // steps_per_epoch
    batch_offset = start_step % steps_per_epoch
    sampler.set_epoch(epoch, start_index=batch_offset * batch_size)
    if resume_path and context.is_main and saved_epoch != epoch:
        print(
            "resume epoch normalized from checkpoint metadata "
            f"{saved_epoch} to step-derived epoch {epoch}"
        )
    data_iterator = iter(loader)
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    run_start = time.perf_counter()

    try:
        for step in range(start_step + 1, max_steps + 1):
            step_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iterator = iter(loader)
                batch = next(data_iterator)

            human = batch["human"].to(
                context.device, non_blocking=True
            )
            robot = batch["robot"].to(
                context.device, non_blocking=True
            )
            task_embeddings = encode_text_batch(text_encoder, batch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            current_lr = scheduler.set_step(step - 1)
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.float16,
                enabled=amp_enabled and context.device.type == "cuda",
            ):
                outputs = model(human, robot, task_embeddings)
                human_global = gather_with_grad(outputs["human_frozen"])
                robot_frozen_global = gather_with_grad(
                    outputs["robot_frozen"]
                )
                robot_adapted_global = gather_with_grad(
                    outputs["robot_adapted"]
                )
                loss, metrics = human_robot_contrastive_loss(
                    human_global,
                    robot_frozen_global,
                    robot_adapted_global,
                    temperature=temperature,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            elapsed = time.perf_counter() - step_start
            adapted_std = robot_adapted_global.float().std(
                dim=0, unbiased=False
            ).mean()
            attention = outputs["robot_adapted_attention"].float()
            attention_entropy = -(
                attention * attention.clamp_min(1e-12).log()
            ).sum(dim=1).mean()

            if step % log_every == 0 or step == 1 or step == max_steps:
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": _mean_metric(loss),
                    **{
                        key: _mean_metric(value)
                        for key, value in metrics.items()
                    },
                    "adapted_embedding_std": _mean_metric(adapted_std),
                    "adapted_attention_entropy": _mean_metric(
                        attention_entropy
                    ),
                    "learning_rate": current_lr,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "seconds": elapsed,
                    "images": actual_global_batch * sampling.num_frames * 3,
                    "cuda_memory_mb": _memory_mb(context.device),
                }
                if context.is_main:
                    assert csv_writer is not None and csv_handle is not None
                    csv_writer.writerow(row)
                    csv_handle.flush()
                    print(
                        f"step {step:05d} loss={row['loss']:.6f} "
                        f"h2r/r2h={row['loss_h2r']:.6f}/{row['loss_r2h']:.6f} "
                        f"top1={row['h2r_top1']:.3f}/{row['r2h_top1']:.3f} "
                        f"std={row['adapted_embedding_std']:.5f} "
                        f"lr={row['learning_rate']:.3e} "
                        f"sec={elapsed:.3f}"
                    )

            if save_every > 0 and step % save_every == 0:
                barrier()
                if context.is_main:
                    save_training_checkpoint(
                        output_dir / f"checkpoint_{step:06d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        step // steps_per_epoch,
                        config,
                    )
                barrier()

        barrier()
        if context.is_main:
            save_training_checkpoint(
                output_dir / "checkpoint_last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                max_steps,
                max_steps // steps_per_epoch,
                config,
            )
            export_official_checkpoint(
                output_dir / "AdaptedR3M_reproduced.pyth",
                model,
                optimizer,
                max_steps,
                steps_per_epoch,
                config,
            )
            print(
                f"done steps={max_steps} "
                f"hours={(time.perf_counter() - run_start) / 3600:.2f} "
                f"output={output_dir}"
            )
        barrier()
    finally:
        if csv_handle is not None:
            csv_handle.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    train(config, args)


if __name__ == "__main__":
    main()
