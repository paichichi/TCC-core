#!/usr/bin/env python3
"""Train a timestamp multi-view TCN triplet model on RH20T groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xirl.models import ViTB16LinearEncoderNet


@dataclass
class ViewRef:
    rel_path: str
    sequence_id: str
    camera_id: int
    frame_idx: int


@dataclass
class Group:
    episode_id: str
    role: str
    timestamp_ms: int
    views: list[ViewRef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcc-root", type=Path, default=Path("/home/paichichi/data/RH20T/TCC_RH20T"))
    parser.add_argument("--index", type=Path, default=Path("/home/paichichi/data/RH20T/TCC_RH20T/tcn_timestamp_groups.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/tcc-core/tcn_triplet_runs/smoke"))
    parser.add_argument("--roles", nargs="+", default=["h", "r"], choices=["h", "r"])
    parser.add_argument("--max-groups", type=int, default=200000)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-triplets-per-batch", type=int, default=32)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrain-path", default="/home/paichichi/data/pretrain/D4R_IN_1M.pth")
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--fusion-size", type=int, default=768)
    parser.add_argument("--num-sample-episodes", type=int, default=0)
    return parser.parse_args()


def load_groups(index_path: Path, roles: set[str], max_groups: int | None) -> dict[tuple[str, str], list[Group]]:
    by_episode_role: dict[tuple[str, str], list[Group]] = defaultdict(list)
    loaded = 0
    with index_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["role"] not in roles:
                continue
            views_raw = json.loads(row["views_json"])
            if len(views_raw) < 2:
                continue
            views = [
                ViewRef(
                    rel_path=view["rel_path"],
                    sequence_id=str(view["sequence_id"]),
                    camera_id=int(view["camera_id"]),
                    frame_idx=int(view["frame_idx"]),
                )
                for view in views_raw
            ]
            group = Group(
                episode_id=row["episode_id"],
                role=row["role"],
                timestamp_ms=int(row["target_timestamp_ms"]),
                views=views,
            )
            by_episode_role[(group.episode_id, group.role)].append(group)
            loaded += 1
            if max_groups is not None and loaded >= max_groups:
                break

    # Keep only episode/role tracks where a future negative exists.
    out = {}
    for key, groups in by_episode_role.items():
        groups.sort(key=lambda group: group.timestamp_ms)
        if len(groups) >= 2:
            out[key] = groups
    return out


def restrict_episode_count(
    tracks: dict[tuple[str, str], list[Group]],
    num_episodes: int,
) -> dict[tuple[str, str], list[Group]]:
    if num_episodes <= 0:
        return tracks
    episode_ids = sorted({episode_id for episode_id, _ in tracks})
    keep = set(episode_ids[:num_episodes])
    return {key: value for key, value in tracks.items() if key[0] in keep}


def make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_image(root: Path, rel_path: str, transform) -> torch.Tensor:
    image = Image.open(root / rel_path).convert("RGB")
    return transform(image)


def sample_triplet_paths(
    tracks: dict[tuple[str, str], list[Group]],
    max_triplets: int,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str], list[int]]:
    keys = list(tracks)
    anchors: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []
    gaps: list[int] = []

    attempts = 0
    while len(anchors) < max_triplets and attempts < max_triplets * 20:
        attempts += 1
        key = rng.choice(keys)
        groups = tracks[key]
        if len(groups) < 2:
            continue
        anchor_idx = rng.randrange(0, len(groups) - 1)
        negative_idx = rng.randrange(anchor_idx + 1, len(groups))
        anchor_group = groups[anchor_idx]
        negative_group = groups[negative_idx]
        if len(anchor_group.views) < 2 or not negative_group.views:
            continue

        view_indices = list(range(len(anchor_group.views)))
        for a_idx, p_idx in combinations_sample(view_indices, rng):
            if len(anchors) >= max_triplets:
                break
            anchor_view = anchor_group.views[a_idx]
            positive_view = anchor_group.views[p_idx]
            negative_view = rng.choice(negative_group.views)
            anchors.append(anchor_view.rel_path)
            positives.append(positive_view.rel_path)
            negatives.append(negative_view.rel_path)
            gaps.append(negative_group.timestamp_ms - anchor_group.timestamp_ms)
    if not anchors:
        raise RuntimeError("Could not sample any TCN triplets.")
    return anchors, positives, negatives, gaps


def combinations_sample(indices: list[int], rng: random.Random):
    pairs = []
    for i, a_idx in enumerate(indices):
        for p_idx in indices[i + 1:]:
            if rng.random() < 0.5:
                pairs.append((a_idx, p_idx))
            else:
                pairs.append((p_idx, a_idx))
    rng.shuffle(pairs)
    return pairs


def build_model(args: argparse.Namespace):
    model = ViTB16LinearEncoderNet(
        embedding_size=args.embedding_size,
        fusion_size=args.fusion_size,
        vvcl_embedding_size=args.embedding_size,
        num_ctx_frames=1,
        normalize_embeddings=True,
        learnable_temp=False,
        pretrain_path=args.pretrain_path,
        vit_weights="none",
        vit_pooling="patch_mean",
        trainable_scope="layernorm_head",
    )
    return model


def trainable_summary(model) -> str:
    total = 0
    trainable = 0
    parts = []
    for name, module in [
        ("backbone", model.backbone),
        ("fusion_head", model.fusion_head),
        ("encoder", model.encoder),
    ]:
        mod_total = sum(p.numel() for p in module.parameters())
        mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += mod_total
        trainable += mod_train
        parts.append(f"{name}={mod_train}/{mod_total}")
    return f"trainable={trainable}/{total} " + " ".join(parts)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[idx])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_groups(args.index, set(args.roles), args.max_groups)
    tracks = restrict_episode_count(tracks, args.num_sample_episodes)
    if not tracks:
        raise RuntimeError("No valid episode/role tracks loaded.")
    num_groups = sum(len(groups) for groups in tracks.values())
    print(f"loaded tracks={len(tracks)} groups={num_groups} roles={args.roles}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = make_transform(args.image_size)
    model = build_model(args).to(device).train()
    print(trainable_summary(model))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    csv_path = args.out_dir / "losses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "loss",
            "d_pos",
            "d_neg",
            "simple_acc",
            "margin_acc",
            "num_triplets",
            "gap_mean_ms",
            "gap_p10_ms",
            "gap_p50_ms",
            "gap_p90_ms",
            "seconds",
            "cuda_mem_mb",
        ])
        f.flush()

        start = time.time()
        for step in range(1, args.max_iters + 1):
            iter_start = time.time()
            anchor_paths, positive_paths, negative_paths, gaps = sample_triplet_paths(
                tracks,
                args.max_triplets_per_batch,
                rng,
            )
            paths = anchor_paths + positive_paths + negative_paths
            frames = torch.stack([
                load_image(args.tcc_root, rel_path, transform)
                for rel_path in paths
            ], dim=0).unsqueeze(1).to(device)

            out = model(frames)
            embs = out.embs[:, 0]
            n = len(anchor_paths)
            anchor = embs[:n]
            positive = embs[n:2 * n]
            negative = embs[2 * n:]
            d_pos = torch.linalg.vector_norm(anchor - positive, dim=-1)
            d_neg = torch.linalg.vector_norm(anchor - negative, dim=-1)
            loss = F.relu(d_pos - d_neg + args.margin).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                simple_acc = (d_pos < d_neg).float().mean()
                margin_acc = ((d_pos + args.margin) < d_neg).float().mean()
            seconds = time.time() - iter_start
            mem_mb = (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if device.type == "cuda"
                else 0.0
            )
            row = [
                step,
                f"{loss.item():.8f}",
                f"{d_pos.mean().item():.8f}",
                f"{d_neg.mean().item():.8f}",
                f"{simple_acc.item():.6f}",
                f"{margin_acc.item():.6f}",
                n,
                f"{sum(gaps) / len(gaps):.2f}",
                f"{percentile(gaps, 0.10):.2f}",
                f"{percentile(gaps, 0.50):.2f}",
                f"{percentile(gaps, 0.90):.2f}",
                f"{seconds:.4f}",
                f"{mem_mb:.1f}",
            ]
            writer.writerow(row)
            if step % args.log_every == 0 or step == 1:
                print(
                    f"step {step:05d} loss={row[1]} d_pos={row[2]} d_neg={row[3]} "
                    f"acc={row[4]} margin_acc={row[5]} triplets={n} "
                    f"gap_ms={row[8]}/{row[9]}/{row[10]} mem={row[12]}MB "
                    f"sec={row[11]}",
                    flush=True,
                )
                f.flush()
            if args.save_every and step % args.save_every == 0:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    args.out_dir / f"checkpoint_{step:06d}.pt",
                )
        print(f"done in {time.time() - start:.1f}s; losses={csv_path}")


if __name__ == "__main__":
    main()
