from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
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
    random_horizontal_flip: bool = True

    @property
    def native_stride(self) -> float:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        return self.sampling_rate * self.source_fps / self.target_fps


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
    max_pairs: int | None = 56000,
    subset_seed: int = 0,
    allowed_tasks: Sequence[str] | None = None,
) -> list[PairRecord]:
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

    if max_pairs is not None and max_pairs > 0 and len(records) > max_pairs:
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
        self.config = config
        self.train = train

    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("Cannot transform an empty clip.")

        if self.train:
            short_side = random.randint(
                self.config.jitter_min, self.config.jitter_max
            )
        else:
            short_side = self.config.jitter_min
        resized = [
            tvf.resize(
                image,
                short_side,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            for image in images
        ]

        _, height, width = tvf.get_dimensions(resized[0])
        crop_size = self.config.crop_size
        if min(height, width) < crop_size:
            scale = math.ceil(crop_size / min(height, width) * short_side)
            resized = [
                tvf.resize(
                    image,
                    scale,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                for image in images
            ]
            _, height, width = tvf.get_dimensions(resized[0])

        if self.train:
            top = random.randint(0, height - crop_size)
            left = random.randint(0, width - crop_size)
            flip = (
                self.config.random_horizontal_flip and random.random() < 0.5
            )
        else:
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
            flip = False

        tensors = []
        for image in resized:
            image = tvf.crop(image, top, left, crop_size, crop_size)
            if flip:
                image = tvf.hflip(image)
            tensor = tvf.to_tensor(image)
            tensor = tvf.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            tensors.append(tensor)
        return torch.stack(tensors)


class RH20TPairDataset(Dataset[dict[str, Any]]):
    """One item is one same-camera human/robot video pair."""

    def __init__(
        self,
        data_root: str | Path,
        lookup_path: str | Path,
        manifest_path: str | Path,
        task_descriptions_path: str | Path,
        sampling: SamplingConfig,
        max_pairs: int | None = 56000,
        subset_seed: int = 0,
        allowed_tasks: Sequence[str] | None = None,
        train: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_root = self.data_root / "train"
        self.sampling = sampling
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

    def _sample_relative_indices(self, num_available: int) -> list[int]:
        count = self.sampling.num_frames
        if num_available < 1:
            raise ValueError("A video must contain at least one frame.")

        if self.sampling.mode == "random_sorted":
            if num_available >= count:
                return sorted(random.sample(range(num_available), count))
            return sorted(random.randrange(num_available) for _ in range(count))
        if self.sampling.mode != "slowfast_clip":
            raise ValueError(f"Unknown sampling mode: {self.sampling.mode}")

        stride = self.sampling.native_stride
        max_start = max(0.0, num_available - 1 - stride * (count - 1))
        start = random.uniform(0.0, max_start) if max_start else 0.0
        return [
            min(num_available - 1, max(0, round(start + index * stride)))
            for index in range(count)
        ]

    def _load_clip(
        self,
        record: PairRecord,
        sequence_id: str,
        start_frame: int,
        num_frames: int,
    ) -> torch.Tensor:
        relative_indices = self._sample_relative_indices(num_frames)
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
        return self.transform(images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        human = self._load_clip(
            record,
            record.human_sequence_id,
            record.human_start_frame,
            record.human_num_frames,
        )
        robot = self._load_clip(
            record,
            record.robot_sequence_id,
            record.robot_start_frame,
            record.robot_num_frames,
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
        random_horizontal_flip=bool(
            config.get("random_horizontal_flip", True)
        ),
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
