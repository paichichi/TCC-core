#!/usr/bin/env python3
"""Evaluate within-pair Human-to-Robot progress retrieval."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--tcc-repo", type=Path, required=True)
    parser.add_argument("--hralign-repo", type=Path, required=True)
    parser.add_argument("--bn-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--r3m-pretrain", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def progress_ranges(boundaries: tuple[int, int, int, int]):
    b1, b2, b3, b4 = boundaries
    return {
        1: (0, b1 - 1),
        2: (b1, b2 - 1),
        3: (b2, b3 - 1),
        4: (b3, b4),
    }


def progress_for_frame(frame: int, boundaries: tuple[int, int, int, int]) -> int:
    for progress, (start, end) in progress_ranges(boundaries).items():
        if start <= frame <= end:
            return progress
    raise ValueError(f"Frame {frame} is outside the annotated action range.")


def make_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_images(paths: list[Path], transform) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images)


class BNAffineEncoder:
    name = "resnet_dual_bn_affine"

    def __init__(self, args: argparse.Namespace, device: torch.device):
        sys.path.insert(0, str(args.tcc_repo))
        from scripts.train_multiview_softdtw import ViewSetAttentionSoftDTW

        checkpoint = torch.load(
            args.bn_checkpoint, map_location="cpu", weights_only=False)
        cfg = checkpoint["args"]
        self.model = ViewSetAttentionSoftDTW(
            embedding_size=int(cfg["embedding_size"]),
            fusion_size=int(cfg["fusion_size"]),
            pretrain_path=str(cfg["pretrain_path"]),
            backbone=str(cfg["backbone"]),
            train_layernorm=bool(cfg["train_backbone_norm_affine"]),
            train_adapters=bool(cfg["train_backbone_adapters"]),
            view_token_dropout=float(cfg["view_token_dropout"]),
            view_token_noise_std=float(cfg["view_token_noise_std"]),
            attention_dropout=float(cfg["attention_dropout"]),
            projector_dropout=float(cfg["projector_dropout"]),
            adapter_domain=str(cfg["adapter_domain"]),
            view_mode=str(cfg["view_mode"]),
            representation_mode=str(cfg["representation_mode"]),
        )
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, images: torch.Tensor, domain: str) -> torch.Tensor:
        del domain
        images = images.to(self.device)
        pooled = self.model.backbone(images)
        pooled = self.model.view_norm(pooled)
        pooled = self.model.pooled_norm(pooled)
        return F.normalize(pooled.float(), dim=-1).cpu()


class R3MAdapterEncoder:
    name = "r3m_align_adapter"

    def __init__(self, args: argparse.Namespace, device: torch.device):
        sys.path.insert(0, str(args.hralign_repo))
        from hralign.models import HRAlignR3ML

        checkpoint = torch.load(
            args.adapter_checkpoint, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]["model"]
        self.model = HRAlignR3ML(
            pretrain_path=args.r3m_pretrain,
            adapted_bn_mode=str(cfg["adapted_bn_mode"]),
            normalize_language_query=bool(cfg["normalize_language_query"]),
            normalize_visual_tokens_for_attention=bool(
                cfg["normalize_visual_tokens_for_attention"]),
            normalize_pooled_features=bool(cfg["normalize_pooled_features"]),
        )
        self.model.load_trainable_state(checkpoint["trainable_state"])
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, images: torch.Tensor, domain: str) -> torch.Tensor:
        images = images.to(self.device)
        base = self.model.adapted.forward_base(images)
        if domain == "robot":
            base = self.model.adapted.apply_adapters(base)
        pooled = torch.flatten(self.model.adapted.convnet.avgpool(base), 1)
        return F.normalize(pooled.float(), dim=-1).cpu()


def encode_paths(encoder, paths, transform, batch_size, domain):
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = load_images(paths[start:start + batch_size], transform)
        chunks.append(encoder.encode(images, domain))
    return torch.cat(chunks)


def build_queries(metadata_rows, annotation_rows, seed):
    metadata = {row["local_pair"]: row for row in metadata_rows}
    annotations = defaultdict(dict)
    for row in annotation_rows:
        annotations[row["local_pair"]][row["domain"]] = tuple(
            int(row[key]) for key in (
                "b1_action_start", "b2_lift", "b3_placed", "b4_leave"))

    rng = random.Random(seed)
    queries = []
    for pair_name in sorted(annotations):
        if pair_name not in metadata:
            raise KeyError(f"No metadata for {pair_name}")
        human = annotations[pair_name]["human"]
        robot = annotations[pair_name]["robot"]
        for progress, (start, end) in progress_ranges(human).items():
            if end < start:
                raise ValueError(f"Empty progress {progress} for {pair_name}")
            queries.append({
                "local_pair": pair_name,
                "episode_id": metadata[pair_name]["episode_id"],
                "human_sequence_id": metadata[pair_name]["human_sequence_id"],
                "robot_sequence_id": metadata[pair_name]["robot_sequence_id"],
                "progress": progress,
                "human_query_frame": rng.randint(start, end),
                "robot_b1": robot[0],
                "robot_b2": robot[1],
                "robot_b3": robot[2],
                "robot_b4": robot[3],
            })
    return queries


def evaluate_encoder(args, encoder, queries, transform):
    rows = []
    grouped = defaultdict(list)
    for query in queries:
        grouped[query["local_pair"]].append(query)

    for pair_name in sorted(grouped):
        pair_queries = grouped[pair_name]
        episode_id = int(pair_queries[0]["episode_id"])
        h_seq = pair_queries[0]["human_sequence_id"]
        r_seq = pair_queries[0]["robot_sequence_id"]
        episode_dir = args.data_root / "train" / f"episode_{episode_id:06d}"
        human_paths = [
            episode_dir / h_seq / f"{int(q['human_query_frame']):06d}.jpg"
            for q in pair_queries
        ]
        robot_b4 = int(pair_queries[0]["robot_b4"])
        robot_paths = [
            episode_dir / r_seq / f"{frame:06d}.jpg"
            for frame in range(robot_b4 + 1)
        ]
        for path in human_paths + robot_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        human_z = encode_paths(
            encoder, human_paths, transform, args.batch_size, "human")
        robot_z = encode_paths(
            encoder, robot_paths, transform, args.batch_size, "robot")
        similarity = human_z @ robot_z.T
        predictions = similarity.argmax(dim=1)

        robot_bounds = tuple(
            int(pair_queries[0][key])
            for key in ("robot_b1", "robot_b2", "robot_b3", "robot_b4"))
        for index, query in enumerate(pair_queries):
            predicted_frame = int(predictions[index])
            predicted_progress = progress_for_frame(predicted_frame, robot_bounds)
            expected_progress = int(query["progress"])
            rows.append({
                "model": encoder.name,
                "local_pair": pair_name,
                "episode_id": episode_id,
                "progress": expected_progress,
                "human_query_frame": int(query["human_query_frame"]),
                "robot_retrieved_frame": predicted_frame,
                "robot_retrieved_progress": predicted_progress,
                "correct": int(predicted_progress == expected_progress),
                "cosine_similarity": f"{float(similarity[index, predicted_frame]):.8f}",
            })
    return rows


def summarize(all_rows):
    summaries = []
    for model in sorted({row["model"] for row in all_rows}):
        model_rows = [row for row in all_rows if row["model"] == model]
        for progress in (1, 2, 3, 4):
            subset = [row for row in model_rows if row["progress"] == progress]
            correct = sum(int(row["correct"]) for row in subset)
            summaries.append({
                "model": model,
                "scope": f"progress_{progress}",
                "correct": correct,
                "queries": len(subset),
                "accuracy": f"{correct / len(subset):.6f}",
            })
        correct = sum(int(row["correct"]) for row in model_rows)
        summaries.append({
            "model": model,
            "scope": "overall",
            "correct": correct,
            "queries": len(model_rows),
            "accuracy": f"{correct / len(model_rows):.6f}",
        })
    return summaries


def main():
    args = parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    transform = make_transform()
    queries = build_queries(
        read_csv(args.metadata), read_csv(args.annotations), args.seed)
    write_csv(args.out_dir / "sampled_queries.csv", queries)

    encoders = [
        R3MAdapterEncoder(args, device),
        BNAffineEncoder(args, device),
    ]
    all_rows = []
    for encoder in encoders:
        rows = evaluate_encoder(args, encoder, queries, transform)
        write_csv(args.out_dir / f"{encoder.name}_retrieval.csv", rows)
        all_rows.extend(rows)
        del encoder
    write_csv(args.out_dir / "summary.csv", summarize(all_rows))


if __name__ == "__main__":
    main()
