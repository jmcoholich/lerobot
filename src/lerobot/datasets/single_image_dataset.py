#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import copy
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

import torch

from lerobot.utils.constants import OBS_IMAGE


class SingleImageDataset(torch.utils.data.Dataset):
    """Expose selected camera frames as independent, single-image samples."""

    def __init__(
        self,
        dataset,
        image_keys: Sequence[str],
        frame_indices: Sequence[int] | None = None,
    ):
        if not image_keys:
            raise ValueError("image_keys must contain at least one camera key")
        if len(set(image_keys)) != len(image_keys):
            raise ValueError("image_keys must not contain duplicates")

        camera_keys = set(dataset.meta.camera_keys)
        missing_keys = set(image_keys).difference(camera_keys)
        if missing_keys:
            raise ValueError(f"Selected image keys are missing from the dataset: {sorted(missing_keys)}")

        shapes = {tuple(dataset.meta.features[key]["shape"]) for key in image_keys}
        if len(shapes) != 1:
            raise ValueError(f"Selected image keys must have matching shapes, got {sorted(shapes)}")

        if frame_indices is None:
            frame_indices = range(len(dataset))
        if any(index < 0 or index >= len(dataset) for index in frame_indices):
            raise IndexError("frame_indices contains an index outside the source dataset")

        self.dataset = dataset
        self.image_keys = tuple(image_keys)
        self.frame_indices = frame_indices
        self._source_camera_keys = camera_keys

        features = {
            key: feature
            for key, feature in dataset.meta.features.items()
            if key not in camera_keys
        }
        image_feature = dict(dataset.meta.features[self.image_keys[0]])
        image_feature["dtype"] = "image"
        features[OBS_IMAGE] = image_feature

        self.meta = copy.copy(dataset.meta)
        self.meta.info = {
            **dataset.meta.info,
            "features": features,
            "total_frames": len(self),
        }
        if dataset.meta.stats is not None:
            self.meta.stats = {
                key: stats
                for key, stats in dataset.meta.stats.items()
                if key not in camera_keys
            }

    def __len__(self) -> int:
        return len(self.frame_indices) * len(self.image_keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        frame_position, camera_position = divmod(index, len(self.image_keys))
        frame_index = self.frame_indices[frame_position]
        image_key = self.image_keys[camera_position]
        item = self.dataset.get_item(frame_index, camera_keys=(image_key,))

        image = item[image_key]
        item = {key: value for key, value in item.items() if key not in self._source_camera_keys}
        item[OBS_IMAGE] = image
        return item

    def get_episode_sample_indices(self, max_episode_steps: int | None = None) -> list[list[int]]:
        """Return this dataset's sample indices grouped by source episode."""
        episode_indices = self.dataset.episodes
        if episode_indices is None:
            episode_indices = range(self.dataset.num_episodes)

        episode_lengths = [self.dataset.meta.episodes[index]["length"] for index in episode_indices]
        episode_ends = []
        total_frames = 0
        for length in episode_lengths:
            total_frames += length
            episode_ends.append(total_frames)
        if total_frames != len(self.dataset):
            raise ValueError(
                f"Episode lengths sum to {total_frames}, but the source dataset has {len(self.dataset)} frames"
            )

        samples_by_episode = [[] for _ in episode_lengths]
        num_cameras = len(self.image_keys)
        for frame_position, frame_index in enumerate(self.frame_indices):
            episode_position = bisect_right(episode_ends, frame_index)
            sample_start = frame_position * num_cameras
            samples_by_episode[episode_position].extend(range(sample_start, sample_start + num_cameras))

        return [
            samples
            for samples, length in zip(samples_by_episode, episode_lengths, strict=True)
            if samples and (max_episode_steps is None or length <= max_episode_steps)
        ]

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return self.dataset.num_episodes

    @property
    def episodes(self):
        return self.dataset.episodes
