from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PairRecord:
    index: int
    episode_id: int
    human_sequence_id: str
    robot_sequence_id: str
    human_start_frame: int
    robot_start_frame: int
    human_num_frames: int
    robot_num_frames: int
    task_id: str
    task_text: str
    camera_id: str


@dataclass(frozen=True)
class SamplingConfig:
    num_frames: int = 5
    mode: str = "slowfast_clip"
    frame_index_mode: str = "compact"
    sampling_rate: float = 12.0
    source_fps: float = 10.0
    target_fps: float = 30.0
    crop_size: int = 224
    jitter_min: int = 256
    jitter_max: int = 320
    spatial_mode: str = "random_resized_crop"
    relative_crop_scale: tuple[float, float] = (0.08, 1.0)
    relative_crop_aspect: tuple[float, float] = (0.75, 1.3333)
    random_horizontal_flip: bool = True

    @property
    def native_clip_size(self) -> float:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        return max(
            1.0,
            self.sampling_rate
            * self.num_frames
            * self.source_fps
            / self.target_fps,
        )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_task_descriptions(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result = {}
    for task_id, description in raw.items():
        if isinstance(description, str):
            result[task_id] = description
        else:
            result[task_id] = description["task_description_english"]
    return result


def load_pair_records(
    lookup_path: str | Path,
    manifest_path: str | Path,
    task_descriptions_path: str | Path,
    max_pairs: int | None = None,
    subset_seed: int = 0,
    allowed_tasks: Sequence[str] | None = None,
) -> list[PairRecord]:
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer or None.")
    manifest_rows = _read_csv(manifest_path)
    sequence_metadata = {row["sequence_id"]: row for row in manifest_rows}
    task_descriptions = _load_task_descriptions(task_descriptions_path)
    allowed = set(allowed_tasks or ())

    records = []
    for row in _read_csv(lookup_path):
        task_id = row["task_id"]
        if allowed and task_id not in allowed:
            continue
        if task_id not in task_descriptions:
            raise KeyError(f"No English task description for {task_id}.")

        human_id = row["human_sequence_id"]
        robot_id = row["robot_sequence_id"]
        try:
            human_meta = sequence_metadata[human_id]
            robot_meta = sequence_metadata[robot_id]
        except KeyError as error:
            raise KeyError(
                f"Sequence {error.args[0]} from lookup is absent from manifest."
            ) from error
        if human_meta["role"] != "h" or robot_meta["role"] != "r":
            raise ValueError(
                f"Invalid roles for pair {human_id}/{robot_id}: "
                f"{human_meta['role']}/{robot_meta['role']}."
            )

        expected_metadata = (
            (
                human_id,
                human_meta,
                robot_id,
                row["human_num_frames"],
            ),
            (
                robot_id,
                robot_meta,
                human_id,
                row["robot_num_frames"],
            ),
        )
        for sequence_id, metadata, paired_id, num_frames in expected_metadata:
            checks = {
                "paired_sequence_id": paired_id,
                "episode_id": row["episode_id"],
                "task_id": task_id,
                "camera_id": row["camera_id"],
                "num_frames": num_frames,
            }
            for field, expected in checks.items():
                actual = metadata.get(field)
                if actual not in (None, "") and str(actual) != str(expected):
                    raise ValueError(
                        f"Manifest mismatch for sequence {sequence_id}: "
                        f"{field}={actual!r}, expected {expected!r}."
                    )

        records.append(
            PairRecord(
                index=len(records),
                episode_id=int(row["episode_id"]),
                human_sequence_id=human_id,
                robot_sequence_id=robot_id,
                human_start_frame=int(human_meta["start_frame"]),
                robot_start_frame=int(robot_meta["start_frame"]),
                human_num_frames=int(row["human_num_frames"]),
                robot_num_frames=int(row["robot_num_frames"]),
                task_id=task_id,
                task_text=task_descriptions[task_id],
                camera_id=row["camera_id"],
            )
        )

    if max_pairs is not None and len(records) > max_pairs:
        generator = random.Random(subset_seed)
        selected = sorted(generator.sample(range(len(records)), max_pairs))
        records = [records[index] for index in selected]

    return [
        PairRecord(**{**record.__dict__, "index": index})
        for index, record in enumerate(records)
    ]


class ClipTransform:
    """Applies one spatial transform consistently to every frame in a clip."""

    def __init__(self, config: SamplingConfig, train: bool = True):
        if config.spatial_mode not in {
            "random_resized_crop",
            "short_side_jitter",
        }:
            raise ValueError(
                f"Unknown spatial mode: {config.spatial_mode}"
            )
        self.config = config
        self.train = train

    def _random_resized_crop(
        self,
        frames: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        height, width = frames.shape[-2:]
        scale = self.config.relative_crop_scale
        ratio = self.config.relative_crop_aspect
        for _ in range(10):
            target_area = rng.uniform(*scale) * height * width
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(rng.uniform(*log_ratio))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = rng.randint(0, height - crop_height)
                left = rng.randint(0, width - crop_width)
                break
        else:
            input_ratio = width / height
            if input_ratio < ratio[0]:
                crop_width = width
                crop_height = int(round(crop_width / ratio[0]))
            elif input_ratio > ratio[1]:
                crop_height = height
                crop_width = int(round(crop_height * ratio[1]))
            else:
                crop_height, crop_width = height, width
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2

        frames = frames[
            :,
            :,
            top : top + crop_height,
            left : left + crop_width,
        ]
        return F.interpolate(
            frames,
            size=(self.config.crop_size, self.config.crop_size),
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _resize_short_side(
        frames: torch.Tensor,
        short_side: int,
    ) -> torch.Tensor:
        height, width = frames.shape[-2:]
        if width < height:
            new_width = short_side
            new_height = math.floor(height / width * short_side)
        else:
            new_height = short_side
            new_width = math.floor(width / height * short_side)
        if (new_height, new_width) == (height, width):
            return frames
        return F.interpolate(
            frames,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

    def __call__(
        self,
        images: list[Image.Image],
        rng: random.Random | None = None,
    ) -> torch.Tensor:
        if not images:
            raise ValueError("Cannot transform an empty clip.")
        rng = rng or random.Random()
        dimensions = {
            tuple(tvf.get_dimensions(image)) for image in images
        }
        if len(dimensions) != 1:
            raise ValueError(
                "All frames in one clip must have identical dimensions."
            )

        frames = torch.stack([tvf.to_tensor(image) for image in images])
        frames = tvf.normalize(frames, IMAGENET_MEAN, IMAGENET_STD)
        crop_size = self.config.crop_size
        if self.train and self.config.spatial_mode == "random_resized_crop":
            frames = self._random_resized_crop(frames, rng)
        else:
            if self.train:
                short_side = int(
                    round(
                        rng.uniform(
                            self.config.jitter_min,
                            self.config.jitter_max,
                        )
                    )
                )
            else:
                short_side = self.config.jitter_min
            short_side = max(short_side, crop_size)
            frames = self._resize_short_side(frames, short_side)
            height, width = frames.shape[-2:]
            if self.train:
                top = rng.randint(0, height - crop_size)
                left = rng.randint(0, width - crop_size)
            else:
                top = (height - crop_size) // 2
                left = (width - crop_size) // 2
            frames = frames[
                :, :, top : top + crop_size, left : left + crop_size
            ]

        if (
            self.train
            and self.config.random_horizontal_flip
            and rng.random() < 0.5
        ):
            frames = frames.flip(-1)
        return frames


class RH20TPairDataset(Dataset[dict[str, Any]]):
    """One item is one same-camera human/robot video pair."""

    def __init__(
        self,
        data_root: str | Path,
        lookup_path: str | Path,
        manifest_path: str | Path,
        task_descriptions_path: str | Path,
        sampling: SamplingConfig,
        max_pairs: int | None = None,
        subset_seed: int = 0,
        allowed_tasks: Sequence[str] | None = None,
        train: bool = True,
        augmentation_seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.image_root = self.data_root / "train"
        self.sampling = sampling
        self.augmentation_seed = int(augmentation_seed)
        self.records = load_pair_records(
            lookup_path=lookup_path,
            manifest_path=manifest_path,
            task_descriptions_path=task_descriptions_path,
            max_pairs=max_pairs,
            subset_seed=subset_seed,
            allowed_tasks=allowed_tasks,
        )
        self.transform = ClipTransform(sampling, train=train)
        if not self.records:
            raise ValueError("No RH20T H/R pairs matched the dataset filters.")

    def __len__(self) -> int:
        return len(self.records)

    def _sample_relative_indices(
        self,
        num_available: int,
        rng: random.Random,
    ) -> list[int]:
        count = self.sampling.num_frames
        if num_available < 1:
            raise ValueError("A video must contain at least one frame.")

        if self.sampling.mode == "random_sorted":
            if num_available >= count:
                return sorted(rng.sample(range(num_available), count))
            return sorted(rng.randrange(num_available) for _ in range(count))
        if self.sampling.mode != "slowfast_clip":
            raise ValueError(f"Unknown sampling mode: {self.sampling.mode}")

        # This mirrors SlowFast's full-video decode path: choose a random
        # clip, then use torch.linspace(...).long() to sample within it.
        clip_size = self.sampling.native_clip_size
        max_start = max(0.0, num_available - clip_size)
        start = rng.uniform(0.0, max_start) if max_start else 0.0
        end = start + clip_size - 1.0
        indices = torch.linspace(start, end, count)
        return (
            indices.clamp(0, num_available - 1)
            .to(dtype=torch.long)
            .tolist()
        )

    def _load_clip(
        self,
        record: PairRecord,
        sequence_id: str,
        start_frame: int,
        num_frames: int,
        rng: random.Random,
    ) -> torch.Tensor:
        relative_indices = self._sample_relative_indices(num_frames, rng)
        episode = f"episode_{record.episode_id:06d}"
        sequence_root = self.image_root / episode / sequence_id
        images = []
        for relative_index in relative_indices:
            if self.sampling.frame_index_mode == "compact":
                frame_index = relative_index
            elif self.sampling.frame_index_mode == "manifest_offset":
                frame_index = start_frame + relative_index
            else:
                raise ValueError(
                    "Unknown frame_index_mode: "
                    f"{self.sampling.frame_index_mode}"
                )
            path = sequence_root / f"{frame_index:06d}.jpg"
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Missing RH20T frame for pair index={record.index}, "
                    f"task={record.task_id}: {path}"
                ) from error
        return self.transform(images, rng)

    def _rng_for(
        self,
        record_index: int,
        epoch: int,
        stream_offset: int,
    ) -> random.Random:
        seed = (
            self.augmentation_seed * 1_000_003
            + epoch * 97_409
            + record_index * 2_003
            + stream_offset
        ) % 2**64
        return random.Random(seed)

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> dict[str, Any]:
        if isinstance(index, tuple):
            index, epoch = index
        else:
            epoch = 0
        record = self.records[index]
        human = self._load_clip(
            record,
            record.human_sequence_id,
            record.human_start_frame,
            record.human_num_frames,
            self._rng_for(record.index, epoch, 0),
        )
        robot = self._load_clip(
            record,
            record.robot_sequence_id,
            record.robot_start_frame,
            record.robot_num_frames,
            self._rng_for(record.index, epoch, 1),
        )
        return {
            "human": human,
            "robot": robot,
            "task_id": record.task_id,
            "task_text": record.task_text,
            "pair_index": record.index,
            "camera_id": record.camera_id,
        }


def sampling_config_from_dict(config: dict[str, Any]) -> SamplingConfig:
    jitter = config.get("train_jitter_scales", [256, 320])
    relative_scale = config.get("relative_crop_scale", [0.08, 1.0])
    relative_aspect = config.get(
        "relative_crop_aspect", [0.75, 1.3333]
    )
    return SamplingConfig(
        num_frames=int(config.get("num_frames", 5)),
        mode=str(config.get("mode", "slowfast_clip")),
        frame_index_mode=str(config.get("frame_index_mode", "compact")),
        sampling_rate=float(config.get("sampling_rate", 12)),
        source_fps=float(config.get("source_fps", 10)),
        target_fps=float(config.get("target_fps", 30)),
        crop_size=int(config.get("crop_size", 224)),
        jitter_min=int(jitter[0]),
        jitter_max=int(jitter[1]),
        spatial_mode=str(
            config.get("spatial_mode", "random_resized_crop")
        ),
        relative_crop_scale=(
            float(relative_scale[0]),
            float(relative_scale[1]),
        ),
        relative_crop_aspect=(
            float(relative_aspect[0]),
            float(relative_aspect[1]),
        ),
        random_horizontal_flip=bool(
            config.get("random_horizontal_flip", True)
        ),
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
