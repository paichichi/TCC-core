#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hralign.data import load_pair_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the RH20T pair index.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-descriptions", required=True)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional deterministic debug subset; omitted means all pairs.",
    )
    parser.add_argument("--global-batch-size", type=int, default=200)
    parser.add_argument(
        "--frame-index-mode",
        choices=("compact", "manifest_offset"),
        default="compact",
        help=(
            "compact checks 000000..N-1 files used by TCC_RH20T; "
            "manifest_offset checks original frame IDs."
        ),
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check first and last JPEG of every selected sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_pair_records(
        args.lookup,
        args.manifest,
        args.task_descriptions,
        max_pairs=args.max_pairs,
    )
    print(f"selected_pairs: {len(records)}")
    print(
        "complete_global_batches: "
        f"{len(records) // args.global_batch_size}"
    )
    print(f"tasks: {len(set(record.task_id for record in records))}")
    print(f"cameras: {len(set(record.camera_id for record in records))}")
    print("top_tasks:")
    for task_id, count in Counter(
        record.task_id for record in records
    ).most_common(15):
        print(f"  {task_id}: {count}")

    if not args.check_files:
        return
    image_root = Path(args.root) / "train"
    missing = []
    for record in records:
        episode = f"episode_{record.episode_id:06d}"
        for sequence_id, start, count in (
            (
                record.human_sequence_id,
                record.human_start_frame,
                record.human_num_frames,
            ),
            (
                record.robot_sequence_id,
                record.robot_start_frame,
                record.robot_num_frames,
            ),
        ):
            root = image_root / episode / sequence_id
            if args.frame_index_mode == "compact":
                boundary_frames = (0, count - 1)
            else:
                boundary_frames = (start, start + count - 1)
            for frame in boundary_frames:
                path = root / f"{frame:06d}.jpg"
                if not path.is_file():
                    missing.append(str(path))
    print(f"missing_boundary_frames: {len(missing)}")
    for path in missing[:20]:
        print(f"  {path}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
