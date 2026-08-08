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

import torch

from lerobot.datasets.single_image_dataset import SingleImageDataset
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

IMAGE_KEYS = ("observation.images.first", "observation.images.second")
TARGET_KEY = "sparse_returns_gamma_1.0"


class _Metadata:
    def __init__(self):
        self.info = {
            "total_frames": 3,
            "total_episodes": 1,
            "features": {
                OBS_STATE: {"dtype": "float32", "shape": (1,), "names": ["state"]},
                IMAGE_KEYS[0]: {
                    "dtype": "video",
                    "shape": (8, 8, 3),
                    "names": ["height", "width", "channel"],
                },
                IMAGE_KEYS[1]: {
                    "dtype": "video",
                    "shape": (8, 8, 3),
                    "names": ["height", "width", "channel"],
                },
                TARGET_KEY: {"dtype": "float32", "shape": (1,), "names": ["return"]},
            },
        }
        self.stats = {key: {"mean": torch.tensor([0.0])} for key in self.info["features"]}

    @property
    def features(self):
        return self.info["features"]

    @property
    def camera_keys(self):
        return [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in ["image", "video"]
        ]


class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.meta = _Metadata()
        self.num_episodes = 1
        self.episodes = None
        self.requests = []

    def __len__(self):
        return 3

    def get_item(self, frame, camera_keys=None):
        self.requests.append((frame, camera_keys))
        return {
            OBS_STATE: torch.tensor([float(frame)]),
            IMAGE_KEYS[0]: torch.full((3, 8, 8), 10.0 + frame),
            IMAGE_KEYS[1]: torch.full((3, 8, 8), 20.0 + frame),
            TARGET_KEY: torch.tensor(float(frame) / 2),
            "task": f"task{frame}",
        }


def test_single_image_dataset_expands_and_canonicalizes_camera_samples():
    source = _Dataset()
    dataset = SingleImageDataset(source, [IMAGE_KEYS[1], IMAGE_KEYS[0]])

    samples = [dataset[index] for index in range(len(dataset))]

    assert len(dataset) == 6
    assert dataset.meta.camera_keys == [OBS_IMAGE]
    assert OBS_IMAGE not in dataset.meta.stats
    assert TARGET_KEY in dataset.meta.stats
    assert all(set(sample).isdisjoint(IMAGE_KEYS) for sample in samples)
    torch.testing.assert_close(
        torch.stack([sample[OBS_IMAGE][0, 0, 0] for sample in samples]),
        torch.tensor([20.0, 10.0, 21.0, 11.0, 22.0, 12.0]),
    )
    torch.testing.assert_close(
        torch.stack([sample[TARGET_KEY] for sample in samples]),
        torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]),
    )
    assert source.requests == [
        (0, (IMAGE_KEYS[1],)),
        (0, (IMAGE_KEYS[0],)),
        (1, (IMAGE_KEYS[1],)),
        (1, (IMAGE_KEYS[0],)),
        (2, (IMAGE_KEYS[1],)),
        (2, (IMAGE_KEYS[0],)),
    ]


def test_single_image_dataset_applies_frame_stride_before_expansion():
    dataset = SingleImageDataset(
        _Dataset(),
        IMAGE_KEYS,
        frame_indices=range(0, 3, 2),
    )

    samples = [dataset[index] for index in range(len(dataset))]

    assert len(samples) == 4
    torch.testing.assert_close(
        torch.stack([sample[OBS_STATE][0] for sample in samples]),
        torch.tensor([0.0, 0.0, 2.0, 2.0]),
    )


def test_shuffled_loader_visits_every_camera_image_once():
    dataset = SingleImageDataset(_Dataset(), IMAGE_KEYS)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )

    values = torch.cat([batch[OBS_IMAGE][:, 0, 0, 0] for batch in loader])

    torch.testing.assert_close(values.sort().values, torch.tensor([10.0, 11.0, 12.0, 20.0, 21.0, 22.0]))
    assert not torch.equal(values, values.sort().values)
