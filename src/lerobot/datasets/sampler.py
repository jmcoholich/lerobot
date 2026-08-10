#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class EpisodeBalancedSampler:
    """Sample within eligible episodes, optionally giving every episode equal probability."""

    def __init__(
        self,
        sample_indices_by_episode: list[list[int]],
        equal_weight_episodes: bool,
        shuffle: bool,
        seed: int,
        reset_seed_each_iteration: bool = False,
    ):
        if not sample_indices_by_episode:
            raise ValueError("No eligible episodes remain after episode filtering")

        self.sample_indices_by_episode = sample_indices_by_episode
        self.indices = [
            index for episode_indices in sample_indices_by_episode for index in episode_indices
        ]
        self.equal_weight_episodes = equal_weight_episodes
        self.shuffle = shuffle
        self.seed = seed
        self.reset_seed_each_iteration = reset_seed_each_iteration
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        generator = (
            torch.Generator().manual_seed(self.seed)
            if self.reset_seed_each_iteration
            else self.generator
        )
        if self.equal_weight_episodes:
            for _ in range(len(self)):
                episode_position = torch.randint(
                    len(self.sample_indices_by_episode), size=(), generator=generator
                ).item()
                episode_samples = self.sample_indices_by_episode[episode_position]
                sample_position = torch.randint(
                    len(episode_samples), size=(), generator=generator
                ).item()
                yield episode_samples[sample_position]
        elif self.shuffle:
            for position in torch.randperm(len(self), generator=generator).tolist():
                yield self.indices[position]
        else:
            yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)
