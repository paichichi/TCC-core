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


class VideoBatchSampler(abc.ABC, Sampler):
  """Base class for all video samplers."""

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
  ):
    """Constructor.

    Args:
      dir_tree: The directory tree of a `datasets.VideoDataset`.
      batch_size: The number of videos in a batch.
      sequential: Set to `True` to disable any shuffling or randomness.
    """
    assert isinstance(batch_size, int)

    self._batch_size = batch_size
    self._dir_tree = dir_tree
    self._sequential = sequential

  @abc.abstractmethod
  def _generate_indices(self):
    """Generate batch chunks containing (class idx, video_idx) tuples."""
    pass

  def __iter__(self):
    idxs = self._generate_indices()
    if self._sequential:
      return iter(idxs)
    return iter(idxs[i] for i in torch.randperm(len(idxs)))

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
    all_idxs = [all_idxs[i] for i in torch.randperm(len(all_idxs))]
    # If we have less total videos than the batch size, we pad with clones
    # until we reach a length of batch_size.
    if len(all_idxs) < self._batch_size:
      while len(all_idxs) < self._batch_size:
        all_idxs.append(all_idxs[np.random.randint(0, len(all_idxs))])
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
        seq = [seq[i] for i in torch.randperm(len(seq))]
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
  """Samples adjacent human-robot pairs with a dynamic frame count.

  The emitted batch order is:
    [role_order[0]_0, role_order[1]_0, role_order[0]_1, role_order[1]_1, ...]

  Each yielded index is `(class_idx, video_idx, num_frames)`. `num_frames` is
  shared by the whole batch and is derived from the shortest sequence in that
  batch.
  """

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
      metadata_path=None,
      sample_ratio=0.5,
      max_frames=40,
      min_frames=16,
      drop_short_pairs=True,
      role_order=("h", "r"),
  ):
    if batch_size < 2 or batch_size % 2:
      raise ValueError("PairedBatchSampler requires an even batch size >= 2.")
    if metadata_path is None:
      raise ValueError("PairedBatchSampler requires a metadata CSV path.")
    if sample_ratio <= 0:
      raise ValueError("sample_ratio must be positive.")

    self._dir_tree = dir_tree
    self._batch_size = batch_size
    self._pairs_per_batch = batch_size // 2
    self._sequential = sequential
    self._metadata_path = pathlib.Path(metadata_path)
    self._sample_ratio = sample_ratio
    self._max_frames = max_frames
    self._min_frames = min_frames
    self._drop_short_pairs = drop_short_pairs
    self._role_order = tuple(role_order)
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

      first_len = int(row["num_frames"])
      second_len = int(paired["num_frames"])
      pair_min_len = min(first_len, second_len)
      if self._drop_short_pairs and pair_min_len < self._min_frames:
        continue

      pairs.append((video_index[first_key], video_index[second_key],
                    pair_min_len))
    return pairs

  def _num_frames_for_batch(self, pairs):
    batch_min_len = min(pair[-1] for pair in pairs)
    num_frames = int(batch_min_len * self._sample_ratio)
    num_frames = min(num_frames, self._max_frames)
    num_frames = max(num_frames, self._min_frames)
    return min(num_frames, batch_min_len)

  def _generate_indices(self):
    pair_idxs = list(range(len(self._pairs)))
    if not self._sequential:
      pair_idxs = [pair_idxs[i] for i in torch.randperm(len(pair_idxs))]

    end = self._pairs_per_batch * (len(pair_idxs) // self._pairs_per_batch)
    batches = []
    for i in range(0, end, self._pairs_per_batch):
      pairs = [self._pairs[idx] for idx in pair_idxs[i:i + self._pairs_per_batch]]
      num_frames = self._num_frames_for_batch(pairs)

      batch = []
      for first_idx, second_idx, _ in pairs:
        batch.append((*first_idx, num_frames))
        batch.append((*second_idx, num_frames))
      batches.append(batch)
    return batches

  def __iter__(self):
    return iter(self._generate_indices())

  def __len__(self):
    return len(self._pairs) // self._pairs_per_batch
