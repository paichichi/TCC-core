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

"""Video samplers for mini-batch creation."""

import abc
import csv
import pathlib
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

ClassIdxVideoIdx = Tuple[int, int]
DirTreeIndices = List[List[ClassIdxVideoIdx]]
VideoBatchIter = Iterator[List[ClassIdxVideoIdx]]


def _effective_num_frames(row):
  start_frame = int(row.get("start_frame", 0) or 0)
  return max(0, int(row["num_frames"]) - start_frame)


def _row_task_id(row):
  return row.get("task_id", "")


class VideoBatchSampler(abc.ABC, Sampler):
  """Base class for all video samplers."""

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
      seed=None,
  ):
    """Constructor.

    Args:
      dir_tree: The directory tree of a `datasets.VideoDataset`.
      batch_size: The number of videos in a batch.
      sequential: Set to `True` to disable any shuffling or randomness.
      seed: Optional seed for reproducible video sampling.
    """
    assert isinstance(batch_size, int)

    self._batch_size = batch_size
    self._dir_tree = dir_tree
    self._sequential = sequential
    self._seed = seed
    self._torch_generator = None
    self._np_generator = None
    if seed is not None:
      self._torch_generator = torch.Generator()
      self._torch_generator.manual_seed(seed)
      self._np_generator = np.random.default_rng(seed)

  def _randperm(self, n):
    return torch.randperm(n, generator=self._torch_generator)

  def _randint(self, high):
    if self._np_generator is None:
      return np.random.randint(0, high)
    return self._np_generator.integers(0, high)

  @abc.abstractmethod
  def _generate_indices(self):
    """Generate batch chunks containing (class idx, video_idx) tuples."""
    pass

  def __iter__(self):
    idxs = self._generate_indices()
    if self._sequential:
      return iter(idxs)
    return iter(idxs[i] for i in self._randperm(len(idxs)))

  def __len__(self):
    num_vids = 0
    for vids in self._dir_tree.values():
      num_vids += len(vids)
    return num_vids // self.batch_size

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def dir_tree(self):
    return self._dir_tree


class RandomBatchSampler(VideoBatchSampler):
  """Randomly samples videos from different classes into the same batch.

  Note the `sequential` arg is disabled here.
  """

  def _generate_indices(self):
    # Generate a list of video indices for every class.
    all_idxs = []
    for k, v in enumerate(self._dir_tree.values()):
      seq = list(range(len(v)))
      all_idxs.extend([(k, s) for s in seq])
    # Shuffle the indices.
    all_idxs = [all_idxs[i] for i in self._randperm(len(all_idxs))]
    # If we have less total videos than the batch size, we pad with clones
    # until we reach a length of batch_size.
    if len(all_idxs) < self._batch_size:
      while len(all_idxs) < self._batch_size:
        all_idxs.append(all_idxs[self._randint(len(all_idxs))])
    # Split the list of indices into chunks of len `batch_size`.
    idxs = []
    end = self._batch_size * (len(all_idxs) // self._batch_size)
    for i in range(0, end, self._batch_size):
      batch_idxs = all_idxs[i:i + self._batch_size]
      idxs.append(batch_idxs)
    return idxs


class SameClassBatchSampler(VideoBatchSampler):
  """Ensures all videos in a batch belong to the same class."""

  def _generate_indices(self):
    idxs = []
    for k, v in enumerate(self._dir_tree.values()):
      # Generate a list of indices for every video in the class.
      len_v = len(v)
      seq = list(range(len_v))
      if not self._sequential:
        seq = [seq[i] for i in self._randperm(len(seq))]
      # Split the list of indices into chunks of len `batch_size`,
      # ensuring we drop the last chunk if it is not of adequate length.
      batch_idxs = []
      end = self._batch_size * (len_v // self._batch_size)
      for i in range(0, end, self._batch_size):
        xs = seq[i:i + self._batch_size]
        # Add the class index to the video index.
        xs = [(k, x) for x in xs]
        batch_idxs.append(xs)
      idxs.extend(batch_idxs)
    return idxs


class PairedBatchSampler(Sampler):
  """Samples adjacent human-robot pairs with a shared frame count.

  The emitted batch order is:
    [role_order[0]_0, role_order[1]_0, role_order[0]_1, role_order[1]_1, ...]

  Each yielded index is `(class_idx, video_idx, num_frames)`. `num_frames` is
  shared by the whole batch. It can be fixed explicitly, or derived from the
  shortest sequence in the batch.
  """

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
      metadata_path=None,
      fixed_frames=-1,
      sample_ratio=0.5,
      max_frames=-1,
      min_frames=16,
      drop_short_pairs=True,
      role_order=("h", "r"),
      distinct_tasks_per_batch=False,
      seed=None,
  ):
    if batch_size < 2 or batch_size % 2:
      raise ValueError("PairedBatchSampler requires an even batch size >= 2.")
    if metadata_path is None:
      raise ValueError("PairedBatchSampler requires a metadata CSV path.")
    if fixed_frames is not None and 0 < fixed_frames < 2:
      raise ValueError("fixed_frames must be -1 or >= 2.")
    if sample_ratio <= 0:
      raise ValueError("sample_ratio must be positive.")

    self._dir_tree = dir_tree
    self._batch_size = batch_size
    self._pairs_per_batch = batch_size // 2
    self._sequential = sequential
    self._metadata_path = pathlib.Path(metadata_path)
    self._fixed_frames = fixed_frames
    self._sample_ratio = sample_ratio
    self._max_frames = max_frames
    self._min_frames = min_frames
    self._drop_short_pairs = drop_short_pairs
    self._role_order = tuple(role_order)
    self._distinct_tasks_per_batch = distinct_tasks_per_batch
    self._seed = seed
    self._torch_generator = None
    if seed is not None:
      self._torch_generator = torch.Generator()
      self._torch_generator.manual_seed(seed)
    self._pairs = self._load_pairs()

    if not self._pairs:
      raise ValueError("No valid human-robot pairs were found.")

  def _video_index(self):
    index = {}
    for class_idx, (class_path, video_paths) in enumerate(
        self._dir_tree.items()):
      task_id = pathlib.Path(class_path).stem
      for video_idx, video_path in enumerate(video_paths):
        sequence_id = pathlib.Path(video_path).stem
        video_index = (class_idx, video_idx)
        index[sequence_id] = video_index
        index[(task_id, sequence_id)] = video_index
    return index

  def _load_pairs(self):
    if not self._metadata_path.exists():
      raise ValueError(f"Metadata CSV not found: {self._metadata_path}")

    video_index = self._video_index()
    with self._metadata_path.open("r", newline="") as fp:
      rows = list(csv.DictReader(fp))
    rows_by_id = {row["sequence_id"]: row for row in rows}

    pairs = []
    first_role, second_role = self._role_order
    for row in rows:
      if row["role"] != first_role:
        continue
      paired = rows_by_id.get(row["paired_sequence_id"])
      if paired is None or paired["role"] != second_role:
        continue
      first_key = row["sequence_id"]
      second_key = paired["sequence_id"]
      if first_key not in video_index or second_key not in video_index:
        continue

      first_len = _effective_num_frames(row)
      second_len = _effective_num_frames(paired)
      pair_min_len = min(first_len, second_len)
      if self._drop_short_pairs and pair_min_len < self._min_frames:
        continue

      pairs.append((video_index[first_key], video_index[second_key],
                    pair_min_len, _row_task_id(row)))
    return pairs

  def _num_frames_for_batch(self, pairs):
    batch_min_len = min(pair[2] for pair in pairs)
    if self._fixed_frames is not None and self._fixed_frames > 0:
      return self._fixed_frames
    num_frames = int(batch_min_len * self._sample_ratio)
    if self._max_frames is not None and self._max_frames > 0:
      num_frames = min(num_frames, self._max_frames)
    num_frames = max(num_frames, self._min_frames)
    return min(num_frames, batch_min_len)

  def _generate_indices(self):
    pair_idxs = list(range(len(self._pairs)))
    if not self._sequential:
      pair_idxs = [
          pair_idxs[i] for i in torch.randperm(
              len(pair_idxs), generator=self._torch_generator)
      ]

    batches = []
    pending = []
    pending_tasks = set()
    for pair_idx in pair_idxs:
      pair = self._pairs[pair_idx]
      task_id = pair[3]
      if (self._distinct_tasks_per_batch and pending and
          task_id in pending_tasks):
        continue

      pending.append(pair)
      pending_tasks.add(task_id)
      if len(pending) < self._pairs_per_batch:
        continue

      pairs = pending
      num_frames = self._num_frames_for_batch(pairs)

      batch = []
      for first_idx, second_idx, _, _ in pairs:
        batch.append((*first_idx, num_frames))
        batch.append((*second_idx, num_frames))
      batches.append(batch)
      pending = []
      pending_tasks = set()
    return batches

  def __iter__(self):
    return iter(self._generate_indices())

  def __len__(self):
    return len(self._pairs) // self._pairs_per_batch


class CrossCameraPairedBatchSampler(Sampler):
  """Samples same-episode pairs from different camera views.

  The emitted batch order is:
    [view_a_0, view_b_0, view_a_1, view_b_1, ...]

  By default, each pair shares task, episode, role, and embodiment, but has
  different camera IDs. If pair_role_order is set, e.g. ("h", "r"), each pair
  uses those two roles while still requiring different camera IDs. Batches
  prefer distinct episodes across pairs, which keeps batch_size=4 as two
  independent cross-camera pairs.
  """

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
      metadata_path=None,
      fixed_frames=-1,
      sample_ratio=0.5,
      max_frames=-1,
      min_frames=16,
      drop_short_pairs=True,
      roles=("h", "r"),
      pair_role_order=(),
      distinct_episodes=True,
      distinct_tasks_per_batch=False,
      seed=None,
  ):
    if batch_size < 2 or batch_size % 2:
      raise ValueError(
          "CrossCameraPairedBatchSampler requires an even batch size >= 2.")
    if metadata_path is None:
      raise ValueError(
          "CrossCameraPairedBatchSampler requires a metadata CSV path.")
    if fixed_frames is not None and 0 < fixed_frames < 2:
      raise ValueError("fixed_frames must be -1 or >= 2.")
    if sample_ratio <= 0:
      raise ValueError("sample_ratio must be positive.")
    if pair_role_order and len(pair_role_order) != 2:
      raise ValueError("pair_role_order must be empty or contain two roles.")

    self._dir_tree = dir_tree
    self._batch_size = batch_size
    self._pairs_per_batch = batch_size // 2
    self._sequential = sequential
    self._metadata_path = pathlib.Path(metadata_path)
    self._fixed_frames = fixed_frames
    self._sample_ratio = sample_ratio
    self._max_frames = max_frames
    self._min_frames = min_frames
    self._drop_short_pairs = drop_short_pairs
    self._roles = set(roles)
    self._pair_role_order = tuple(pair_role_order)
    self._distinct_episodes = distinct_episodes
    self._distinct_tasks_per_batch = distinct_tasks_per_batch
    self._seed = seed
    self._torch_generator = None
    if seed is not None:
      self._torch_generator = torch.Generator()
      self._torch_generator.manual_seed(seed)
    self._pairs = self._load_pairs()

    if not self._pairs:
      raise ValueError("No valid cross-camera pairs were found.")

  def _video_index(self):
    index = {}
    for class_idx, (class_path, video_paths) in enumerate(
        self._dir_tree.items()):
      task_id = pathlib.Path(class_path).stem
      for video_idx, video_path in enumerate(video_paths):
        sequence_id = pathlib.Path(video_path).stem
        video_index = (class_idx, video_idx)
        index[sequence_id] = video_index
        index[(task_id, sequence_id)] = video_index
    return index

  def _indexed_rows(self, video_index):
    with self._metadata_path.open("r", newline="") as fp:
      rows = list(csv.DictReader(fp))
    return [
        row for row in rows
        if row["role"] in self._roles and row["sequence_id"] in video_index
    ]

  def _make_pair(self, first, second, video_index, episode_id, task_id):
    if first["camera_id"] == second["camera_id"]:
      return None

    first_len = _effective_num_frames(first)
    second_len = _effective_num_frames(second)
    pair_min_len = min(first_len, second_len)
    if self._drop_short_pairs and pair_min_len < self._min_frames:
      return None

    return (
        video_index[first["sequence_id"]],
        video_index[second["sequence_id"]],
        pair_min_len,
        episode_id,
        task_id,
    )

  def _load_same_role_pairs(self, rows, video_index):
    groups = {}
    for row in rows:
      key = (
          row["task_id"],
          row["episode_id"],
          row["role"],
          row["embodiment_id"],
      )
      groups.setdefault(key, []).append(row)

    pairs = []
    for key, group_rows in groups.items():
      group_rows = sorted(group_rows, key=lambda row: int(row["camera_id"]))
      for first_pos in range(len(group_rows)):
        for second_pos in range(first_pos + 1, len(group_rows)):
          first = group_rows[first_pos]
          second = group_rows[second_pos]
          pair = self._make_pair(first, second, video_index, key[1], key[0])
          if pair is not None:
            pairs.append(pair)
    return pairs

  def _load_cross_role_pairs(self, rows, video_index):
    first_role, second_role = self._pair_role_order
    groups = {}
    for row in rows:
      key = (
          row["task_id"],
          row["episode_id"],
          row["embodiment_id"],
      )
      groups.setdefault(key, []).append(row)

    pairs = []
    for key, group_rows in groups.items():
      first_rows = sorted(
          [row for row in group_rows if row["role"] == first_role],
          key=lambda row: int(row["camera_id"]))
      second_rows = sorted(
          [row for row in group_rows if row["role"] == second_role],
          key=lambda row: int(row["camera_id"]))
      for first in first_rows:
        for second in second_rows:
          pair = self._make_pair(first, second, video_index, key[1], key[0])
          if pair is not None:
            pairs.append(pair)
    return pairs

  def _load_pairs(self):
    if not self._metadata_path.exists():
      raise ValueError(f"Metadata CSV not found: {self._metadata_path}")

    video_index = self._video_index()
    rows = self._indexed_rows(video_index)
    if self._pair_role_order:
      return self._load_cross_role_pairs(rows, video_index)
    return self._load_same_role_pairs(rows, video_index)

  def _num_frames_for_batch(self, pairs):
    batch_min_len = min(pair[2] for pair in pairs)
    if self._fixed_frames is not None and self._fixed_frames > 0:
      return self._fixed_frames
    num_frames = int(batch_min_len * self._sample_ratio)
    if self._max_frames is not None and self._max_frames > 0:
      num_frames = min(num_frames, self._max_frames)
    num_frames = max(num_frames, self._min_frames)
    return min(num_frames, batch_min_len)

  def _shuffled_pair_indices(self):
    pair_idxs = list(range(len(self._pairs)))
    if self._sequential:
      return pair_idxs
    return [
        pair_idxs[i] for i in torch.randperm(
            len(pair_idxs), generator=self._torch_generator)
    ]

  def _generate_indices(self):
    pair_idxs = self._shuffled_pair_indices()
    batches = []
    pending = []
    pending_episodes = set()
    pending_tasks = set()

    for pair_idx in pair_idxs:
      pair = self._pairs[pair_idx]
      episode_id = pair[3]
      task_id = pair[4]
      if (self._distinct_episodes and pending and
          episode_id in pending_episodes):
        continue
      if (self._distinct_tasks_per_batch and pending and
          task_id in pending_tasks):
        continue

      pending.append(pair)
      pending_episodes.add(episode_id)
      pending_tasks.add(task_id)

      if len(pending) == self._pairs_per_batch:
        num_frames = self._num_frames_for_batch(pending)
        batch = []
        for first_idx, second_idx, _, _, _ in pending:
          batch.append((*first_idx, num_frames))
          batch.append((*second_idx, num_frames))
        batches.append(batch)
        pending = []
        pending_episodes = set()
        pending_tasks = set()

    return batches

  def __iter__(self):
    return iter(self._generate_indices())

  def __len__(self):
    if self._distinct_episodes:
      episode_ids = {pair[3] for pair in self._pairs}
      if len(episode_ids) < self._pairs_per_batch:
        return 0
    if self._distinct_tasks_per_batch:
      task_ids = {pair[4] for pair in self._pairs}
      if len(task_ids) < self._pairs_per_batch:
        return 0
    if len(self._pairs) < self._pairs_per_batch:
      return 0
    return len(self._pairs) // self._pairs_per_batch


class TaskRoleCameraBatchSampler(Sampler):
  """Samples role pairs from the same task with different camera views.

  Unlike CrossCameraPairedBatchSampler, this sampler does not require pairs to
  come from the same episode and does not materialize all possible pairs. It
  builds role-specific pools per metadata group and samples pairs online.
  """

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
      metadata_path=None,
      fixed_frames=-1,
      sample_ratio=0.5,
      max_frames=-1,
      min_frames=16,
      drop_short_pairs=True,
      group_keys=("task_id",),
      pair_role_order=("h", "r"),
      different_camera=True,
      different_episode=False,
      distinct_tasks_per_batch=False,
      seed=None,
  ):
    if batch_size < 2 or batch_size % 2:
      raise ValueError("TaskRoleCameraBatchSampler requires an even batch.")
    if metadata_path is None:
      raise ValueError("TaskRoleCameraBatchSampler requires a metadata CSV.")
    if fixed_frames is not None and 0 < fixed_frames < 2:
      raise ValueError("fixed_frames must be -1 or >= 2.")
    if sample_ratio <= 0:
      raise ValueError("sample_ratio must be positive.")
    if len(pair_role_order) != 2:
      raise ValueError("pair_role_order must contain two roles.")
    if not group_keys:
      raise ValueError("group_keys must contain at least one metadata key.")

    self._dir_tree = dir_tree
    self._batch_size = batch_size
    self._pairs_per_batch = batch_size // 2
    self._sequential = sequential
    self._metadata_path = pathlib.Path(metadata_path)
    self._fixed_frames = fixed_frames
    self._sample_ratio = sample_ratio
    self._max_frames = max_frames
    self._min_frames = min_frames
    self._drop_short_pairs = drop_short_pairs
    self._group_keys = tuple(group_keys)
    self._pair_role_order = tuple(pair_role_order)
    self._different_camera = different_camera
    self._different_episode = different_episode
    self._distinct_tasks_per_batch = distinct_tasks_per_batch
    self._seed = seed
    self._torch_generator = None
    if seed is not None:
      self._torch_generator = torch.Generator()
      self._torch_generator.manual_seed(seed)

    self._groups = self._load_groups()
    if not self._groups:
      raise ValueError("No valid task role/camera groups were found.")

  def _video_index(self):
    index = {}
    for class_idx, (class_path, video_paths) in enumerate(
        self._dir_tree.items()):
      task_id = pathlib.Path(class_path).stem
      for video_idx, video_path in enumerate(video_paths):
        sequence_id = pathlib.Path(video_path).stem
        video_index = (class_idx, video_idx)
        index[sequence_id] = video_index
        index[(task_id, sequence_id)] = video_index
    return index

  def _randint(self, high):
    return int(torch.randint(
        high, (1,), generator=self._torch_generator).item())

  def _load_groups(self):
    if not self._metadata_path.exists():
      raise ValueError(f"Metadata CSV not found: {self._metadata_path}")

    video_index = self._video_index()
    first_role, second_role = self._pair_role_order
    groups = {}
    with self._metadata_path.open("r", newline="") as fp:
      for row in csv.DictReader(fp):
        if row["sequence_id"] not in video_index:
          continue
        if row["role"] not in self._pair_role_order:
          continue
        if self._drop_short_pairs and _effective_num_frames(row) < self._min_frames:
          continue
        key = tuple(row[group_key] for group_key in self._group_keys)
        row = dict(row)
        row["_video_index"] = video_index[row["sequence_id"]]
        groups.setdefault(key, {first_role: [], second_role: []})
        groups[key][row["role"]].append(row)

    valid_groups = []
    for key, role_rows in groups.items():
      first_rows = role_rows[first_role]
      second_rows = role_rows[second_role]
      if not first_rows or not second_rows:
        continue
      if not self._has_valid_pair(first_rows, second_rows):
        continue
      valid_groups.append((key, first_rows, second_rows))
    return valid_groups

  def _has_valid_pair(self, first_rows, second_rows):
    for first in first_rows:
      for second in second_rows:
        if self._valid_pair(first, second):
          return True
    return False

  def _valid_pair(self, first, second):
    if self._different_camera and first["camera_id"] == second["camera_id"]:
      return False
    if self._different_episode and first["episode_id"] == second["episode_id"]:
      return False
    return True

  def _sample_pair(self):
    for _ in range(100):
      _, first_rows, second_rows = self._groups[self._randint(len(self._groups))]
      first = first_rows[self._randint(len(first_rows))]
      second = second_rows[self._randint(len(second_rows))]
      if self._valid_pair(first, second):
        return first, second

    _, first_rows, second_rows = self._groups[self._randint(len(self._groups))]
    candidates = [
        (first, second)
        for first in first_rows
        for second in second_rows
        if self._valid_pair(first, second)
    ]
    return candidates[self._randint(len(candidates))]

  def _sample_batch_pairs(self):
    pairs = []
    task_ids = set()
    max_attempts = max(100, 20 * self._pairs_per_batch)
    for _ in range(max_attempts):
      first, second = self._sample_pair()
      task_id = _row_task_id(first)
      if self._distinct_tasks_per_batch and task_id in task_ids:
        continue
      pairs.append((first, second))
      task_ids.add(task_id)
      if len(pairs) == self._pairs_per_batch:
        return pairs
    raise ValueError(
        "Unable to sample a task-distinct batch. Reduce batch size or disable "
        "distinct_tasks_per_batch.")

    _, first_rows, second_rows = self._groups[self._randint(len(self._groups))]
    candidates = [
        (first, second)
        for first in first_rows
        for second in second_rows
        if self._valid_pair(first, second)
    ]
    return candidates[self._randint(len(candidates))]

  def _num_frames_for_batch(self, pairs):
    batch_min_len = min(
        min(_effective_num_frames(first), _effective_num_frames(second))
        for first, second in pairs)
    if self._fixed_frames is not None and self._fixed_frames > 0:
      return self._fixed_frames
    num_frames = int(batch_min_len * self._sample_ratio)
    if self._max_frames is not None and self._max_frames > 0:
      num_frames = min(num_frames, self._max_frames)
    num_frames = max(num_frames, self._min_frames)
    return min(num_frames, batch_min_len)

  def _generate_indices(self):
    batches = []
    for _ in range(len(self)):
      pairs = self._sample_batch_pairs()
      num_frames = self._num_frames_for_batch(pairs)
      batch = []
      for first, second in pairs:
        batch.append((*first["_video_index"], num_frames))
        batch.append((*second["_video_index"], num_frames))
      batches.append(batch)
    return batches

  def __iter__(self):
    return iter(self._generate_indices())

  def __len__(self):
    num_rows = 0
    for _, first_rows, second_rows in self._groups:
      num_rows += len(first_rows) + len(second_rows)
    return max(1, num_rows // self._batch_size)
