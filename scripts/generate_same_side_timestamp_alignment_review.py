#!/usr/bin/env python3
"""Generate a small HTML review for same-side multi-view timestamp alignment."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CameraData:
    name: str
    video_path: Path
    timestamps: np.ndarray
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, default=Path("/mnt/g/RH20T/RGB"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/same_side_timestamp_alignment_review"),
    )
    parser.add_argument("--cfg", default="RH20T_cfg1")
    parser.add_argument("--num-episodes-per-side", type=int, default=3)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--num-timepoints", type=int, default=8)
    parser.add_argument("--max-delta-ms", type=float, default=33.0)
    parser.add_argument("--image-width", type=int, default=320)
    return parser.parse_args()


def read_frame(video_path: Path, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def load_camera(cam_dir: Path) -> CameraData | None:
    color_dir = cam_dir / "color"
    video_path = color_dir / "color.mp4"
    ts_path = color_dir / "timestamps.npy"
    if not video_path.exists() or not ts_path.exists():
        return None
    timestamps = np.load(ts_path)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    usable = min(frame_count, len(timestamps))
    if usable <= 0:
        return None
    return CameraData(
        name=cam_dir.name,
        video_path=video_path,
        timestamps=timestamps[:usable].astype(np.int64),
        frame_count=usable,
    )


def nearest_index(timestamps: np.ndarray, target_ts: int) -> int:
    pos = int(np.searchsorted(timestamps, target_ts))
    candidates = []
    if pos < len(timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    return min(candidates, key=lambda idx: abs(int(timestamps[idx]) - target_ts))


def resize_with_label(
    image: Image.Image,
    width: int,
    label_lines: list[str],
    ok: bool,
) -> Image.Image:
    ratio = width / image.width
    height = int(round(image.height * ratio))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_h = 58
    canvas = Image.new("RGB", (width, height + label_h), (18, 18, 18))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    bar_color = (25, 120, 75) if ok else (165, 65, 45)
    draw.rectangle([0, height, width, height + label_h], fill=bar_color)
    font = ImageFont.load_default()
    y = height + 6
    for line in label_lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 16
    return canvas


def list_episode_dirs(cfg_dir: Path) -> tuple[list[Path], list[Path]]:
    robot = []
    human = []
    for path in sorted(cfg_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("task_"):
            continue
        if path.name.endswith("_human"):
            human.append(path)
        else:
            robot.append(path)
    return robot, human


def paired_episode_dirs(cfg_dir: Path, limit: int) -> list[tuple[str, Path]]:
    robot_dirs, human_dirs = list_episode_dirs(cfg_dir)
    human_by_base = {p.name.removesuffix("_human"): p for p in human_dirs}
    selected: list[tuple[str, Path]] = []
    for robot_dir in robot_dirs:
        if len(selected) >= limit * 2:
            break
        human_dir = human_by_base.get(robot_dir.name)
        if human_dir is None:
            continue
        if len([s for s, _ in selected if s == "human"]) < limit:
            selected.append(("human", human_dir))
        if len([s for s, _ in selected if s == "robot"]) < limit:
            selected.append(("robot", robot_dir))
    return selected


def choose_cameras(episode_dir: Path, num_cameras: int) -> list[CameraData]:
    cams = []
    for cam_dir in sorted(episode_dir.glob("cam_*")):
        cam = load_camera(cam_dir)
        if cam is not None:
            cams.append(cam)
    cams.sort(key=lambda c: c.name)
    return cams[:num_cameras]


def evenly_spaced_targets(start: int, end: int, count: int) -> list[int]:
    if count <= 1:
        return [(start + end) // 2]
    margin = max(1, int((end - start) * 0.06))
    lo = start + margin
    hi = end - margin
    if hi <= lo:
        lo, hi = start, end
    return [int(round(v)) for v in np.linspace(lo, hi, count)]


def max_nearest_delta(cameras: list[CameraData], target_ts: int) -> int:
    return max(
        abs(int(cam.timestamps[nearest_index(cam.timestamps, target_ts)]) - target_ts)
        for cam in cameras
    )


def valid_aligned_targets(
    cameras: list[CameraData],
    overlap_start: int,
    overlap_end: int,
    count: int,
    max_delta_ms: float,
) -> list[int]:
    """Pick target timestamps that are actually supported by every camera."""
    ref = max(cameras, key=lambda cam: cam.frame_count)
    ref_ts = ref.timestamps[(ref.timestamps >= overlap_start) & (ref.timestamps <= overlap_end)]
    valid = [
        int(ts)
        for ts in ref_ts
        if max_nearest_delta(cameras, int(ts)) <= max_delta_ms
    ]
    if len(valid) <= count:
        return valid

    desired = evenly_spaced_targets(valid[0], valid[-1], count)
    picked: list[int] = []
    for target in desired:
        ts = min(valid, key=lambda v: abs(v - target))
        if ts not in picked:
            picked.append(ts)

    # Fill any accidental duplicates with additional spread-out valid timestamps.
    if len(picked) < count:
        for idx in np.linspace(0, len(valid) - 1, count * 2).round().astype(int):
            ts = valid[int(idx)]
            if ts not in picked:
                picked.append(ts)
            if len(picked) >= count:
                break
    return sorted(picked[:count])


def make_review(args: argparse.Namespace) -> None:
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)

    cfg_dir = args.rgb_root / args.cfg
    episodes = paired_episode_dirs(cfg_dir, args.num_episodes_per_side)
    records = []

    for sample_idx, (side, episode_dir) in enumerate(episodes, start=1):
        cameras = choose_cameras(episode_dir, args.num_cameras)
        if len(cameras) < 2:
            continue
        overlap_start = max(int(cam.timestamps[0]) for cam in cameras)
        overlap_end = min(int(cam.timestamps[-1]) for cam in cameras)
        targets = valid_aligned_targets(
            cameras,
            overlap_start,
            overlap_end,
            args.num_timepoints,
            args.max_delta_ms,
        )
        if not targets:
            continue
        sample_id = f"{sample_idx:02d}_{side}_{episode_dir.name}"
        sample_rows = []

        for row_idx, target_ts in enumerate(targets):
            cells = []
            for cam in cameras:
                frame_idx = nearest_index(cam.timestamps, target_ts)
                frame_ts = int(cam.timestamps[frame_idx])
                delta_ms = frame_ts - target_ts
                ok = abs(delta_ms) <= args.max_delta_ms
                image = read_frame(cam.video_path, frame_idx)
                label_lines = [
                    cam.name.replace("cam_", "cam "),
                    f"frame {frame_idx} / {cam.frame_count - 1}",
                    f"delta {delta_ms:+d} ms",
                ]
                labeled = resize_with_label(image, args.image_width, label_lines, ok)
                rel_path = Path("frames") / f"{sample_id}_t{row_idx:02d}_{cam.name}.jpg"
                labeled.save(out_dir / rel_path, quality=88)
                cells.append(
                    {
                        "camera": cam.name,
                        "frame_idx": int(frame_idx),
                        "timestamp": frame_ts,
                        "delta_ms": int(delta_ms),
                        "ok": ok,
                        "image": rel_path.as_posix(),
                    }
                )
            sample_rows.append(
                {
                    "target_timestamp": int(target_ts),
                    "elapsed_ms": int(target_ts - overlap_start),
                    "max_abs_delta_ms": max(abs(c["delta_ms"]) for c in cells),
                    "cells": cells,
                }
            )

        records.append(
            {
                "sample_id": sample_id,
                "side": side,
                "episode": episode_dir.name,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_duration_ms": overlap_end - overlap_start,
                "cameras": [cam.name for cam in cameras],
                "rows": sample_rows,
            }
        )

    (out_dir / "metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_html(out_dir, records, args.max_delta_ms)


def write_html(out_dir: Path, records: list[dict], max_delta_ms: float) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Same-side timestamp alignment review</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f6f7f8;color:#161616}",
        "h1{font-size:24px;margin:0 0 8px}",
        "h2{font-size:18px;margin:28px 0 8px}",
        ".meta{color:#555;margin:4px 0 12px}",
        ".row{margin:14px 0 22px;padding:12px;background:white;border:1px solid #ddd;border-radius:8px}",
        ".rowhead{font-weight:700;margin-bottom:8px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}",
        "img{width:100%;height:auto;display:block;border-radius:4px}",
        ".ok{color:#19713f}.bad{color:#a23b2f}",
        "code{background:#eee;padding:2px 5px;border-radius:4px}",
        "</style></head><body>",
        "<h1>Same-side Multi-view Timestamp Alignment Review</h1>",
        f"<div class='meta'>Nearest timestamp alignment. Green label means |delta| <= {max_delta_ms:.1f} ms.</div>",
    ]
    for rec in records:
        parts.append(f"<h2>{html.escape(rec['sample_id'])}</h2>")
        parts.append(
            "<div class='meta'>"
            f"side=<code>{html.escape(rec['side'])}</code> "
            f"episode=<code>{html.escape(rec['episode'])}</code> "
            f"overlap={rec['overlap_duration_ms'] / 1000:.3f}s "
            f"cameras={', '.join(html.escape(c) for c in rec['cameras'])}"
            "</div>"
        )
        for idx, row in enumerate(rec["rows"]):
            cls = "ok" if row["max_abs_delta_ms"] <= max_delta_ms else "bad"
            parts.append("<div class='row'>")
            parts.append(
                f"<div class='rowhead'>t{idx:02d}: elapsed={row['elapsed_ms'] / 1000:.3f}s, "
                f"target_ts={row['target_timestamp']}, "
                f"max_abs_delta=<span class='{cls}'>{row['max_abs_delta_ms']} ms</span></div>"
            )
            parts.append("<div class='grid'>")
            for cell in row["cells"]:
                parts.append(
                    f"<img src='{html.escape(cell['image'])}' "
                    f"alt='{html.escape(cell['camera'])} frame {cell['frame_idx']}'>"
                )
            parts.append("</div></div>")
    parts.append("</body></html>")
    (out_dir / "preview.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    make_review(parse_args())
