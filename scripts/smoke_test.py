#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from hralign.data import (
    RH20TPairDataset,
    SamplingConfig,
    seed_worker,
)
from hralign.losses import human_robot_contrastive_loss
from hralign.models import HRAlignR3ML
from hralign.text import FrozenR3MTextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One backward pass without downloading DistilBERT."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-descriptions", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="Run the released R3M DistilBERT instead of random text features.",
    )
    parser.add_argument("--tokenizer-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    dataset = RH20TPairDataset(
        data_root=args.root,
        lookup_path=args.lookup,
        manifest_path=args.manifest,
        task_descriptions_path=args.task_descriptions,
        sampling=SamplingConfig(
            num_frames=2,
            sampling_rate=3,
            crop_size=96,
            jitter_min=96,
            jitter_max=112,
        ),
        max_pairs=16,
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    batch = next(iter(loader))
    device = torch.device(args.device)
    model = HRAlignR3ML(args.pretrain).to(device).train()
    human = batch["human"].to(device)
    robot = batch["robot"].to(device)
    if args.with_text:
        text_encoder = FrozenR3MTextEncoder(
            model.source_language_state,
            cache_dir=args.tokenizer_cache_dir,
            local_files_only=args.local_files_only,
        ).to(device)
        task_embeddings = text_encoder(batch["task_text"])
    else:
        task_embeddings = torch.randn(2, 768, device=device)

    start = time.perf_counter()
    output = model(human, robot, task_embeddings)
    loss, metrics = human_robot_contrastive_loss(
        output["human_frozen"],
        output["robot_frozen"],
        output["robot_adapted"],
    )
    loss.backward()
    elapsed = time.perf_counter() - start
    gradients = {
        name: parameter.grad.abs().mean().item()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    print(f"loss: {loss.item():.6f}")
    print(
        "top1 h2r/r2h: "
        f"{metrics['h2r_top1'].item():.3f}/"
        f"{metrics['r2h_top1'].item():.3f}"
    )
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"parameters_with_gradient: {len(gradients)}")
    for name, value in gradients.items():
        print(f"  {name}: {value:.3e}")


if __name__ == "__main__":
    main()
