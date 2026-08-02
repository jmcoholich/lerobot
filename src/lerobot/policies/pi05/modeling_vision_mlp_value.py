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

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from torch import Tensor, nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from lerobot.policies.pi05.configuration_pi05 import PI05Config


_VISION_WEIGHT_PREFIXES = (
    "model.paligemma_with_expert.paligemma.model.vision_tower.",
    "paligemma_with_expert.paligemma.model.vision_tower.",
    "model.vision_tower.",
    "vision_tower.",
)


def _map_vision_weight_key(key: str) -> str | None:
    for prefix in _VISION_WEIGHT_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return None


def _resolve_repo_file(pretrained_path: str, filename: str) -> str | None:
    local_path = Path(pretrained_path)
    if local_path.is_dir():
        candidate = local_path / filename
        return str(candidate) if candidate.is_file() else None
    cached_path = cached_file(
        pretrained_path,
        filename,
        local_files_only=True,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if cached_path is not None:
        return cached_path
    return cached_file(
        pretrained_path,
        filename,
        _raise_exceptions_for_missing_entries=False,
    )


def _load_vision_weights(vision_tower: nn.Module, pretrained_path: str) -> None:
    """Selectively load only SigLIP tensors from a PI0.5 or PaliGemma checkpoint."""
    expected_keys = set(vision_tower.state_dict())
    checkpoint_path = Path(pretrained_path)
    index_path = None if checkpoint_path.is_file() else _resolve_repo_file(
        pretrained_path, SAFE_WEIGHTS_INDEX_NAME
    )

    source_keys_by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if index_path is not None:
        with open(index_path) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        for source_key, shard_name in weight_map.items():
            target_key = _map_vision_weight_key(source_key)
            if target_key in expected_keys:
                source_keys_by_file[shard_name].append((source_key, target_key))
    else:
        if checkpoint_path.is_file():
            weights_path = str(checkpoint_path)
        else:
            weights_path = _resolve_repo_file(pretrained_path, SAFE_WEIGHTS_NAME)
        if weights_path is None:
            raise FileNotFoundError(f"No safetensors checkpoint found at {pretrained_path}")
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                target_key = _map_vision_weight_key(source_key)
                if target_key in expected_keys:
                    source_keys_by_file[weights_path].append((source_key, target_key))

    state_dict = {}
    for shard_name, key_pairs in source_keys_by_file.items():
        shard_path = shard_name
        if index_path is not None:
            shard_path = _resolve_repo_file(pretrained_path, shard_name)
            if shard_path is None:
                raise FileNotFoundError(f"Missing checkpoint shard {shard_name} at {pretrained_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
            for source_key, target_key in key_pairs:
                state_dict[target_key] = checkpoint.get_tensor(source_key)

    missing_keys = sorted(expected_keys - state_dict.keys())
    if missing_keys:
        raise RuntimeError(
            f"Vision checkpoint {pretrained_path} is missing {len(missing_keys)} tensors: "
            f"{missing_keys[:5]}"
        )
    vision_tower.load_state_dict(state_dict, strict=True, assign=True)

    # position_ids is a non-persistent SigLIP buffer, so it is not populated by load_state_dict.
    embeddings = vision_tower.vision_model.embeddings
    if embeddings.position_ids.device.type == "meta":
        embeddings.position_ids = torch.arange(embeddings.num_positions).expand((1, -1))


class PI05VisionMLPValuePytorch(nn.Module):
    """Value model containing only a pretrained SigLIP tower and a compact MLP."""

    def __init__(self, config: PI05Config, load_pretrained: bool = True):
        super().__init__()
        self.config = config
        self.image_resolution = config.image_resolution
        self.num_image_features = len(config.image_features)
        if self.num_image_features == 0:
            raise ValueError("vision_mlp requires at least one image feature")
        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, got {config.image_resolution}"
            )

        vision_config = CONFIG_MAPPING["paligemma"]().vision_config
        vision_config.image_size = config.image_resolution[0]
        vision_config.intermediate_size = 4304
        vision_config.projection_dim = 2048
        vision_config.projector_hidden_act = "gelu_fast"
        vision_config.torch_dtype = "float32"

        pretrained_path = config.vision_encoder_pretrained_path if load_pretrained else None
        if pretrained_path is None:
            self.vision_tower = SiglipVisionModel(vision_config)
        else:
            # Avoid allocating a randomly initialized 412M-parameter tower before assigning its weights.
            with torch.device("meta"):
                self.vision_tower = SiglipVisionModel(vision_config)
            _load_vision_weights(self.vision_tower, pretrained_path)
        self._set_vision_precision(config.dtype)

        vision_dim = self.vision_tower.config.hidden_size
        projection_dim = config.vision_mlp_projection_dim
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, projection_dim),
            nn.SiLU(),
        )

        mlp_input_dim = self.num_image_features * projection_dim + config.max_state_dim
        self.value_mlp = nn.Sequential(
            nn.LayerNorm(mlp_input_dim),
            nn.Linear(mlp_input_dim, config.vision_mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.vision_mlp_dropout),
            nn.Linear(config.vision_mlp_hidden_dim, projection_dim),
            nn.SiLU(),
            nn.Dropout(config.vision_mlp_dropout),
        )
        self.value_head = nn.Linear(projection_dim, config.value_dim)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self._set_requires_grad()

    def _set_vision_precision(self, precision: str) -> None:
        if precision == "bfloat16":
            self.vision_tower.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.vision_tower.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        float32_parameters = (
            "vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.patch_embedding.bias",
            "vision_model.embeddings.position_embedding.weight",
        )
        for name, parameter in self.vision_tower.named_parameters():
            if name in float32_parameters:
                parameter.data = parameter.data.to(dtype=torch.float32)

    def _set_requires_grad(self) -> None:
        if self.config.freeze_vision_encoder or self.config.train_expert_only:
            self.vision_tower.eval()
            self.vision_tower.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder or self.config.train_expert_only:
            self.vision_tower.eval()
        return self

    def gradient_checkpointing_enable(self) -> None:
        self.vision_tower.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.vision_tower.gradient_checkpointing_disable()

    def _drop_input(self, tensor: Tensor) -> Tensor:
        if self.config.input_dropout_percent <= 0 or not self.training:
            return tensor
        keep_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        keep = (
            torch.rand(keep_shape, device=tensor.device)
            >= self.config.input_dropout_percent / 100.0
        )
        return tensor * keep.to(dtype=tensor.dtype)

    def embed_image(self, image: Tensor, image_mask: Tensor) -> Tensor:
        pixel_dtype = self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        image_hidden = self.vision_tower(pixel_values=image.to(dtype=pixel_dtype)).last_hidden_state
        pooled_image = image_hidden.float().mean(dim=1)
        projected_image = self._drop_input(self.vision_projection(pooled_image))
        return projected_image * image_mask[:, None].to(dtype=projected_image.dtype)

    def predict_values(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
    ) -> Tensor:
        if len(images) != self.num_image_features or len(img_masks) != self.num_image_features:
            raise ValueError(
                f"Expected {self.num_image_features} image features, got "
                f"{len(images)} images and {len(img_masks)} masks"
            )

        image_features = [
            self.embed_image(image, image_mask)
            for image, image_mask in zip(images, img_masks, strict=True)
        ]
        if self.config.drop_proprioception_input:
            state = torch.zeros_like(state)
        state = self._drop_input(state.to(dtype=torch.float32))
        features = torch.cat([*image_features, state], dim=-1)
        return self.value_head(self.value_mlp(features))

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        target_values: Tensor,
    ) -> Tensor:
        predictions = self.predict_values(images, img_masks, state)
        return F.mse_loss(predictions, target_values, reduction="none")
