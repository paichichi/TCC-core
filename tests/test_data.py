import csv
import json
from unittest.mock import Mock

from PIL import Image
import pytest

from hralign.data import (
    RH20TPairDataset,
    SamplingConfig,
    load_pair_records,
)
from hralign.trainer import EpochDistributedSampler


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_honors_manifest_start_frames(tmp_path):
    root = tmp_path / "rh20t"
    episode = root / "train" / "episode_000000"
    for sequence, start, color in (
        ("000000", 0, (255, 0, 0)),
        ("000001", 10, (0, 255, 0)),
    ):
        sequence_root = episode / sequence
        sequence_root.mkdir(parents=True)
        for frame in range(start, start + 3):
            Image.new("RGB", (48, 40), color).save(
                sequence_root / f"{frame:06d}.jpg"
            )

    lookup = tmp_path / "lookup.csv"
    write_csv(
        lookup,
        [
            "episode_id",
            "robot_sequence_id",
            "human_sequence_id",
            "task_id",
            "camera_id",
            "robot_num_frames",
            "human_num_frames",
        ],
        [
            {
                "episode_id": "0",
                "robot_sequence_id": "000000",
                "human_sequence_id": "000001",
                "task_id": "task_0001",
                "camera_id": "0",
                "robot_num_frames": "3",
                "human_num_frames": "3",
            }
        ],
    )
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        ["sequence_id", "role", "start_frame"],
        [
            {"sequence_id": "000000", "role": "r", "start_frame": "0"},
            {"sequence_id": "000001", "role": "h", "start_frame": "10"},
        ],
    )
    descriptions = tmp_path / "tasks.json"
    descriptions.write_text(
        json.dumps(
            {
                "task_0001": {
                    "task_description_english": "Press the button"
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = RH20TPairDataset(
        root,
        lookup,
        manifest,
        descriptions,
        SamplingConfig(
            num_frames=2,
            mode="random_sorted",
            frame_index_mode="manifest_offset",
            crop_size=32,
            jitter_min=32,
            jitter_max=32,
            random_horizontal_flip=False,
        ),
        max_pairs=None,
        train=False,
    )
    item = dataset[0]
    assert item["human"].shape == (2, 3, 32, 32)
    assert item["robot"].shape == (2, 3, 32, 32)
    assert item["task_text"] == "Press the button"
    assert item["camera_id"] == "0"


def test_dataset_reads_compact_reindexed_frames(tmp_path):
    root = tmp_path / "rh20t"
    episode = root / "train" / "episode_000000"
    for sequence in ("000000", "000001"):
        sequence_root = episode / sequence
        sequence_root.mkdir(parents=True)
        for frame in range(3):
            Image.new("RGB", (40, 40), (frame * 20, 0, 0)).save(
                sequence_root / f"{frame:06d}.jpg"
            )

    lookup = tmp_path / "lookup.csv"
    write_csv(
        lookup,
        [
            "episode_id",
            "robot_sequence_id",
            "human_sequence_id",
            "task_id",
            "camera_id",
            "robot_num_frames",
            "human_num_frames",
        ],
        [
            {
                "episode_id": "0",
                "robot_sequence_id": "000000",
                "human_sequence_id": "000001",
                "task_id": "task_0001",
                "camera_id": "0",
                "robot_num_frames": "3",
                "human_num_frames": "3",
            }
        ],
    )
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        ["sequence_id", "role", "start_frame"],
        [
            {"sequence_id": "000000", "role": "r", "start_frame": "20"},
            {"sequence_id": "000001", "role": "h", "start_frame": "10"},
        ],
    )
    descriptions = tmp_path / "tasks.json"
    descriptions.write_text(
        json.dumps({"task_0001": "Press the button"}),
        encoding="utf-8",
    )

    dataset = RH20TPairDataset(
        root,
        lookup,
        manifest,
        descriptions,
        SamplingConfig(
            num_frames=2,
            mode="random_sorted",
            frame_index_mode="compact",
            crop_size=32,
            jitter_min=32,
            jitter_max=32,
            random_horizontal_flip=False,
        ),
        max_pairs=None,
        train=False,
    )
    item = dataset[0]
    assert item["human"].shape == (2, 3, 32, 32)
    assert item["robot"].shape == (2, 3, 32, 32)


def test_slowfast_sampling_matches_full_video_decoder_geometry():
    dataset = object.__new__(RH20TPairDataset)
    dataset.sampling = SamplingConfig(
        num_frames=5,
        sampling_rate=12,
        source_fps=10,
        target_fps=30,
    )
    rng = Mock()
    rng.uniform.return_value = 0.0
    assert dataset._sample_relative_indices(100, rng) == [0, 4, 9, 14, 19]


def test_epoch_sampler_resumes_without_repeating_indices():
    sampler = EpochDistributedSampler(
        list(range(8)),
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=7,
        drop_last=True,
    )
    sampler.set_epoch(3)
    full_epoch = list(sampler)
    sampler.set_epoch(3, start_index=4)
    assert list(sampler) == full_epoch[4:]
    assert all(epoch == 3 for _, epoch in full_epoch)


def test_pair_augmentation_rng_is_epoch_and_stream_addressable():
    dataset = object.__new__(RH20TPairDataset)
    dataset.augmentation_seed = 11

    first = dataset._rng_for(17, 4, 0).random()
    repeated = dataset._rng_for(17, 4, 0).random()
    other_epoch = dataset._rng_for(17, 5, 0).random()
    other_stream = dataset._rng_for(17, 4, 1).random()

    assert first == repeated
    assert first != other_epoch
    assert first != other_stream


def test_pair_loader_rejects_manifest_pair_mismatch(tmp_path):
    lookup = tmp_path / "lookup.csv"
    write_csv(
        lookup,
        [
            "episode_id",
            "robot_sequence_id",
            "human_sequence_id",
            "task_id",
            "camera_id",
            "robot_num_frames",
            "human_num_frames",
        ],
        [
            {
                "episode_id": "0",
                "robot_sequence_id": "000000",
                "human_sequence_id": "000001",
                "task_id": "task_0001",
                "camera_id": "0",
                "robot_num_frames": "3",
                "human_num_frames": "3",
            }
        ],
    )
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [
            "sequence_id",
            "paired_sequence_id",
            "episode_id",
            "task_id",
            "role",
            "start_frame",
            "num_frames",
            "camera_id",
        ],
        [
            {
                "sequence_id": "000000",
                "paired_sequence_id": "wrong",
                "episode_id": "0",
                "task_id": "task_0001",
                "role": "r",
                "start_frame": "0",
                "num_frames": "3",
                "camera_id": "0",
            },
            {
                "sequence_id": "000001",
                "paired_sequence_id": "000000",
                "episode_id": "0",
                "task_id": "task_0001",
                "role": "h",
                "start_frame": "0",
                "num_frames": "3",
                "camera_id": "0",
            },
        ],
    )
    descriptions = tmp_path / "tasks.json"
    descriptions.write_text(
        json.dumps({"task_0001": "Press the button"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest mismatch"):
        load_pair_records(
            lookup,
            manifest,
            descriptions,
            max_pairs=None,
        )


def test_pair_loader_rejects_nonpositive_subset_size():
    with pytest.raises(ValueError, match="positive integer or None"):
        load_pair_records(
            "unused_lookup.csv",
            "unused_manifest.csv",
            "unused_tasks.json",
            max_pairs=0,
        )
