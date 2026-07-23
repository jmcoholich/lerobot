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

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText

from lerobot.policies.pi05.configuration_pi05 import PI05Config


class SmolVLMValuePytorch(nn.Module):
    """Scalar value model backed by pretrained SmolVLM image and transformer layers."""

    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config
        torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.smolvlm_pretrained_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        hidden_size = self.vlm.config.text_config.hidden_size
        backbone_dtype = next(self.vlm.model.text_model.parameters()).dtype
        self.image_resolution = (self.vlm.config.vision_config.image_size,) * 2
        self.state_proj = nn.Linear(config.max_state_dim, hidden_size).to(dtype=backbone_dtype)
        self.value_tokens = nn.Parameter(
            torch.randn(config.value_dim, hidden_size, dtype=backbone_dtype) * 0.02
        )
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self._set_requires_grad()

    def _set_requires_grad(self):
        # Language tokens are deliberately omitted for this value model.
        for param in self.vlm.model.text_model.get_input_embeddings().parameters():
            param.requires_grad = False
        for param in self.vlm.lm_head.parameters():
            param.requires_grad = False

        if self.config.freeze_vision_encoder:
            self.vlm.model.vision_model.eval()
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False
        if self.config.train_expert_only:
            self.vlm.eval()
            for param in self.vlm.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vlm.model.vision_model.eval()
        if self.config.train_expert_only:
            self.vlm.eval()
        return self

    def gradient_checkpointing_enable(self):
        self.vlm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.vlm.gradient_checkpointing_disable()

    def _drop_input(self, tensor: Tensor, dropped_value: float = 0.0) -> Tensor:
        if self.config.input_dropout_percent <= 0 or not self.training:
            return tensor
        keep_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        keep = (
            torch.rand(keep_shape, device=tensor.device)
            >= self.config.input_dropout_percent / 100.0
        )
        return torch.where(keep, tensor, dropped_value)

    def embed_image(self, image: Tensor) -> Tensor:
        vision_model = self.vlm.model.vision_model
        image = self._drop_input(image, dropped_value=-1.0).to(
            dtype=next(vision_model.parameters()).dtype
        )
        image_hidden_states = vision_model(pixel_values=image).last_hidden_state
        connector = self.vlm.model.connector
        image_hidden_states = image_hidden_states.to(dtype=next(connector.parameters()).dtype)
        return connector(image_hidden_states)

    def predict_values(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
    ) -> Tensor:
        embs = []
        pad_masks = []

        for image, image_mask in zip(images, img_masks, strict=True):
            image_emb = self.embed_image(image)
            image_emb = image_emb * math.sqrt(image_emb.shape[-1])
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(image_emb.shape[:2]))

        state = self._drop_input(state)
        if self.config.drop_proprioception_input:
            state = torch.zeros_like(state)
        state_emb = self.state_proj(state.to(dtype=self.state_proj.weight.dtype))[:, None, :]
        embs.append(state_emb)
        pad_masks.append(torch.ones(state_emb.shape[:2], dtype=torch.bool, device=state.device))

        value_emb = self.value_tokens[None].expand(state.shape[0], -1, -1)
        embs.append(value_emb)
        pad_masks.append(torch.ones(value_emb.shape[:2], dtype=torch.bool, device=state.device))

        text_dtype = next(self.vlm.model.text_model.parameters()).dtype
        inputs_embeds = torch.cat(embs, dim=1).to(dtype=text_dtype)
        attention_mask = torch.cat(pad_masks, dim=1)
        hidden_states = self.vlm.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        value_hidden = hidden_states[:, -self.config.value_dim :].to(dtype=torch.float32)
        return self.value_head(value_hidden).squeeze(-1)

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        target_values: Tensor,
    ) -> Tensor:
        predictions = self.predict_values(images, img_masks, state)
        return F.mse_loss(predictions, target_values, reduction="none")
