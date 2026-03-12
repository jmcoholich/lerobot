#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import math
from collections import deque
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import cv2
import numpy as np

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available
from lerobot.policies.pi05.action_primitives import get_guidance_action_from_text, get_consistency_guidance, LABEL2ACTION

assert _transformers_available, "Transformers library is required for modeling_pi05.py"
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers.modeling_utils import no_init_weights
from transformers.utils import cached_file
from sklearn.metrics.pairwise import cosine_similarity

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)
import sys
import copy
from functools import partial

from safetensors.torch import load_file
from .vlm_client import VLMClient
from PIL import Image, ImageDraw, ImageFont
# ssh -N -f -L localhost:38477:localhost:38477 -J jcoholich3@sky1.cc.gatech.edu jcoholich3@optimistprime.cc.gatech.edu
VLLM_SERVERS=(
    "http://optimistprime.cc.gatech.edu:38477",
    "http://clippy.cc.gatech.edu:56749",
    "http://shakey.cc.gatech.edu:53727",
    "http://cheetah.cc.gatech.edu:33793",
    "http://ig-88.cc.gatech.edu:56151",
)

TRAJ_COLORS = (
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 0),    # Green
    (255, 255, 255),# White
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (19, 69, 139),  # Brown
)
TRAJ_COLOR_NAMES = (
    "Red",
    "Orange",
    "Blue",
    "Cyan",
    "Magenta",
    "Green",
    "White",
    "Yellow",
    "Purple",
    "Brown",
)

VLM_IO_OUTPUT_DIR = "vlm_io"

# clear the dir if it exists
vlm_io_path = Path(VLM_IO_OUTPUT_DIR)
if vlm_io_path.exists() and vlm_io_path.is_dir():
    for file in vlm_io_path.iterdir():
        if file.is_file():
            file.unlink()

# INTERVENTIONS = False
# INTERVENTIONS = "ensemble"
# INTERVENTIONS = "PIVOT"
INTERVENTIONS = "ensemble"
TRAJ_STD_PERTURB = 0.01  # 0.01
USE_WRIST = False
MANUAL_GUIDANCE = False
VIS_SPREADS = False # no guidance, just generate 5 trajectories and visualize them
if VIS_SPREADS:
    assert not MANUAL_GUIDANCE
    assert not INTERVENTIONS
import random


def color2idx(color):
    """Maps a color string like 'blue' to TRAJ_COLOR_NAMES index """
    try:
        return TRAJ_COLOR_NAMES.index(color.capitalize())
    except ValueError:
        raise ValueError(f"Color '{color}' is not in the list of valid trajectory colors.")


def format_natural_language_list(items: list[str]) -> str:
    """Format items like 'a, b, and c' for prompt text."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def format_small_number_word(value: int) -> str:
    """Format numbers below 10 as words for prompt text."""
    number_words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return number_words.get(value, str(value))


def save_VLM_io(
    pil_img: Image.Image,
    generated_text: str,
    count: int,
    prompt_text: str | None = None,
    output_dir: str | Path = VLM_IO_OUTPUT_DIR,
    suffix=None,
) -> Path:
    """Save an image with VLM prompt text above and output text below."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = pil_img.convert("RGB")
    font = ImageFont.load_default()
    margin = 12
    line_spacing = 4
    max_text_width = max(1, image.width - (2 * margin))

    measure_draw = ImageDraw.Draw(image)
    def _wrap_text(text: str) -> list[str]:
        lines: list[str] = []
        for paragraph in text.splitlines() or [""]:
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            words = paragraph.split()
            current_line = words[0]
            for word in words[1:]:
                candidate = f"{current_line} {word}"
                if measure_draw.textlength(candidate, font=font) <= max_text_width:
                    current_line = candidate
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
        return lines

    prompt_body = prompt_text.strip() if prompt_text else ""
    prompt_combined = "Prompt:" if not prompt_body else f"Prompt:\n{prompt_body}"
    prompt_lines = _wrap_text(prompt_combined)

    output_header = f"Count: {count}"
    output_body = generated_text.strip() if generated_text else ""
    output_combined = output_header if not output_body else f"{output_header}\n{output_body}"
    output_lines = _wrap_text(output_combined)

    line_bbox = measure_draw.textbbox((0, 0), "Ag", font=font)
    line_height = max(1, line_bbox[3] - line_bbox[1])
    top_text_height = (2 * margin) + (len(prompt_lines) * line_height) + (max(0, len(prompt_lines) - 1) * line_spacing)
    bottom_text_height = (2 * margin) + (len(output_lines) * line_height) + (max(0, len(output_lines) - 1) * line_spacing)
    canvas = Image.new("RGB", (image.width, top_text_height + image.height + bottom_text_height), color=(255, 255, 255))
    canvas.paste(image, (0, top_text_height))

    draw = ImageDraw.Draw(canvas)
    y_pos = margin
    for line in prompt_lines:
        draw.text((margin, y_pos), line, fill=(0, 0, 0), font=font)
        y_pos += line_height + line_spacing

    y_pos = top_text_height + image.height + margin
    for line in output_lines:
        draw.text((margin, y_pos), line, fill=(0, 0, 0), font=font)
        y_pos += line_height + line_spacing

    save_path = output_path / f"vlm_io_{count:06d}{f'_{suffix}' if suffix else ''}.png"
    canvas.save(save_path, format="PNG")
    return save_path


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config):
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        num_samples: int = 1,
        guidance_actions: torch.FloatTensor = None,
        guidance_scale: float = 1.0,
        gripper_guidance: bool = True,
        consistency_guidance: torch.FloatTensor = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = num_samples
        device = tokens.device
        use_guidance = guidance_actions is not None and guidance_scale > 0.0
        if use_guidance:
            guidance_actions = guidance_actions.to(device=device, dtype=torch.float32)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # Sample multiple action trajectories if num_samples > 1
        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        if prefix_pad_masks.shape[0] == 1 and bsize > 1:
            # .expand() creates a view without copying memory
            prefix_pad_masks = prefix_pad_masks.expand(bsize, *prefix_pad_masks.shape[1:])
        if past_key_values is not None and bsize > 1:
            # Check if the cache batch size needs expanding (looking at the first layer's keys)
            if past_key_values.key_cache[0].shape[0] == 1:
                expanded_cache = copy.copy(past_key_values)

                # Expand batch dimension for keys and values across all layers
                expanded_cache.key_cache = [
                    k.expand(bsize, *k.shape[1:]) for k in past_key_values.key_cache
                ]
                expanded_cache.value_cache = [
                    v.expand(bsize, *v.shape[1:]) for v in past_key_values.value_cache
                ]

                past_key_values = expanded_cache
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Apply Reconstruction Guidance
            if use_guidance:
                # pi05 convention: x_t = t*noise + (1-t)*data, v_t = noise - data, t: 1→0
                # Clean estimate: x_t - t*v_t  (maps from working model's x_t + (1-t)*v_t via t_pi05=1-t_work, v_pi05=-v_work)
                t_coef = expanded_time.reshape(expanded_time.shape[0], 1, 1)
                clean_x_t_hat = x_t - t_coef * v_t
                residual = clean_x_t_hat[..., :guidance_actions.shape[-1]] - guidance_actions
                residual = F.pad(
                    residual, (0, 24),
                    mode="constant",
                    value=0.0,
                    ) # [bsz, H, A]
                # Analytic gradient: dL/dv_t = -t * residual  (v_t sign flip gives negative of working model's (1-t)*residual)
                grad_vel = -t_coef * residual  # same shape as action_vel

                # Disable guidance on the last action dimension (e.g., gripper)
                if not gripper_guidance:
                    grad_vel[..., -1] = 0.0 # Disable right gripper guidance
                    grad_vel[..., 6] = 0.0 # Disable left gripper guidance

                # Apply guidance: v_t = v_t - scale * (-t * residual) = v_t + scale * t * residual
                v_t = v_t - guidance_scale * grad_vel
            elif consistency_guidance is not None:
                consistency_guidance = consistency_guidance.to(device=device, dtype=torch.float32)
                # pi05 convention: x_t = t*noise + (1-t)*data, v_t = noise - data, t: 1→0
                # Clean estimate: x_t - t*v_t  (maps from working model's x_t + (1-t)*v_t via t_pi05=1-t_work, v_pi05=-v_work)
                t_coef = expanded_time.reshape(expanded_time.shape[0], 1, 1)
                clean_x_t_hat = x_t - t_coef * v_t
                chunk_size = clean_x_t_hat.shape[1]
                residual = clean_x_t_hat[:, 0:1, :7] - consistency_guidance  # exclude gripper, only guide first action in chunk
                residual = F.pad(
                    residual, (0, 25, 0, chunk_size - 1),
                    mode="constant",
                    value=0.0,
                    ) # [bsz, H, A]
                # Analytic gradient: dL/dv_t = -t * residual  (v_t sign flip gives negative of working model's (1-t)*residual)
                grad_vel = -t_coef * residual  # same shape as action_vel

                # Apply guidance: v_t = v_t - scale * (-t * residual) = v_t + scale * t * residual
                v_t = v_t - 40.0 * grad_vel  # hardcode guidance scale to 40
            x_t = x_t + dt * v_t
            time += dt

        return x_t # [batch_size, horizon_steps, action_dim]

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions_and_get_feature(self, images, img_masks, tokens, masks, noise=None, num_steps=None) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, feaure = self.denoise_step_and_getfeature(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t, feaure

    @torch.no_grad()
    def add_and_denoise1step_get_feature(self, images, img_masks, tokens, masks, clean_action, noise=None, num_steps=None) -> Tensor:
        """Do a single denoising and compute the action and feature."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise * (1 / num_steps) + clean_action * ((num_steps - 1) / num_steps)
        time = torch.tensor(1.0, dtype=torch.float32, device=device) + dt * (num_steps - 1)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, feaure = self.denoise_step_and_getfeature(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t, feaure


    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step_and_getfeature(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        feature = suffix_out[:, 0].clone()
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out), feature



class PI05PolicyTaco(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.model = PI05Pytorch(config)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.count = 0

        self.reset()
        if INTERVENTIONS:
            self.vlm_client = VLMClient(server_url="http://127.0.0.1:38477")
        if USE_WRIST:
            prompt_template_path = "src/lerobot/policies/pi05/vlm_prompt_template_wrist.txt"
        else:
            prompt_template_path = "src/lerobot/policies/pi05/vlm_prompt_template.txt"
        primitive_prompt_template_path = "src/lerobot/policies/pi05/primitive_prompt_template.txt"
        with open(prompt_template_path, "r") as f:
            self.pivot_prompt_template = f.readlines()
        with open(primitive_prompt_template_path, "r") as f:
            self.primitive_prompt_template = f.readlines()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        with no_init_weights():
            model = cls(config, **kwargs)
        model.model.paligemma_with_expert.paligemma.tie_weights()

        # Try to load the pytorch_model.bin or model.safetensors file
        print(f"Loading model from: {pretrained_name_or_path}")
        resolved_file = cached_file(
            pretrained_name_or_path,
            "model.safetensors",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download", False),
            resume_download=kwargs.get("resume_download"),
            proxies=kwargs.get("proxies"),
            use_auth_token=kwargs.get("use_auth_token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
        )

        original_state_dict = load_file(resolved_file)
        print("[OK] Loaded state dict from model.safetensors")

        # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
        fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

        # Then add "model." prefix for all keys that don't already have it
        remapped_state_dict = {}
        remap_count = 0

        for key, value in fixed_state_dict.items():
            if not key.startswith("model."):
                new_key = f"model.{key}"
                remapped_state_dict[new_key] = value
                remap_count += 1
                if remap_count <= 10:  # Only print first 10 to avoid spam
                    print(f"Remapped: {key} -> {new_key}")
            else:
                remapped_state_dict[key] = value

        if remap_count > 0:
            print(f"Remapped {remap_count} state dict keys")

        # Load the remapped state dict into the model
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

        if missing_keys != ['model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'] or unexpected_keys:
            raise RuntimeError(f"Unexpected missing or unexpected keys: missing={missing_keys}, unexpected={unexpected_keys}")
        print("All keys loaded successfully!")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def gen_pivot_text_prompt(self, task, num_traj):
        trajectory_choices = ["<TRAJECTORY_CHOICES>"]
        colors = [color_name.lower() for color_name in TRAJ_COLOR_NAMES[:num_traj]]
        for color in colors:
            trajectory_choices.append(
                f'    "{color}": Choose this to command the robot to follow the {color} path.'
            )
        # trajectory_choices.append(
        #     '    "none": Choose this to reject all proposed trajectories. This is the safest option if all paths lead to failure (e.g., collision, incorrect placement).'
        # )
        trajectory_choices.append("</TRAJECTORY_CHOICES>")

        prompt = "".join(copy.copy(self.pivot_prompt_template)).replace("<TASK_DESCRIPTION/>", task)
        prompt = prompt.replace("<TRAJECTORY_CHOICES>", "\n".join(trajectory_choices))
        prompt = prompt.replace("<NUM_TRAJECTORIES/>", format_small_number_word(num_traj))
        prompt = prompt.replace("<TRAJECTORY_COLOR_LIST/>", format_natural_language_list(colors))
        prompt = prompt.replace("<TRAJECTORY_COLOR_OPTIONS/>", ", ".join(f'"{color}"' for color in colors))
        return prompt

    def gen_primitive_text_prompt(self, task):
        prompt = "".join(copy.copy(self.primitive_prompt_template)).replace("<TASK_DESCRIPTION/>", task)
        return prompt

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], postprocessor=None, robot=None) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()


        # guidance_action_1 = get_guidance_action_from_text("up", postprocessor=postprocessor, robot=robot)
        # guidance_action_2 = get_guidance_action_from_text("right", postprocessor=postprocessor, robot=robot)
        # guidance_action_3 = get_guidance_action_from_text("backward", postprocessor=postprocessor, robot=robot)
        # guidance_action = torch.cat([guidance_action_1, guidance_action_2, guidance_action_3], dim=0)
        # import pickle
        # with open("input_args_4.pkl", "wb") as f:
        #     pickle.dump((batch['observation.images.camera_front'], postprocessor, guidance_action), f)
        # breakpoint()
        # visualize_trajectories_on_camera(
        #     batch['observation.images.camera_front'],
        #     guidance_action,
        #     actions_are_normalized=True,
        #     postprocessor=postprocessor,
        #     )
        # guidance_action = None
        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            if self.count == 4 and not MANUAL_GUIDANCE:
                sys.exit(0)
            intervention_period = 1
            # consistency_guidance_action = get_consistency_guidance(postprocessor=postprocessor, robot=robot)
            guidance_scale = 40.0
            print(f"\nGenerating action chunk number {self.count}")
            if self.count % intervention_period == 0 and INTERVENTIONS and not (MANUAL_GUIDANCE or VIS_SPREADS):  # intervene
                print(f"Intervention step...")
                print(f"Intervention mode: {INTERVENTIONS}")
                if INTERVENTIONS == "PIVOT":
                    print("Running pivot guidance...")
                    guidance_action = self.pivot(batch, postprocessor, robot, save_imgs=False)
                elif INTERVENTIONS == "primitive":
                    assert not USE_WRIST
                    print("Running primitive guidance...")
                    guidance_action = self.primitive_guidance(batch, postprocessor, robot, save_imgs=False)
                elif INTERVENTIONS == "ensemble":
                    assert not USE_WRIST
                    print("Running ensemble guidance (pivot + primitive + fusion)...")
                    pivot_guidance_action = self.pivot(batch, postprocessor, robot, save_imgs=False)
                    primitive_guidance_action = self.primitive_guidance(batch, postprocessor, robot, save_imgs=False)
                    guidance_action = self.action_ensemble(pivot_guidance_action, primitive_guidance_action, batch, postprocessor, robot, save_imgs=True)
                else:
                    raise RuntimeError
                actions = self.predict_action_chunk(
                    batch,
                    num_samples=1,
                    guidance_actions=guidance_action,
                    guidance_scale=guidance_scale,
                    consistency_guidance=None,
                    )
            elif MANUAL_GUIDANCE:
                print(f"Manual guidance step...")
                # Example of manual guidance: guide to move right
                guidance_action = None
                x = partial(get_guidance_action_from_text, postprocessor=postprocessor, robot=robot)
                breakpoint()
                actions = self.predict_action_chunk(
                    batch,
                    num_samples=1,
                    guidance_actions=guidance_action,
                    guidance_scale=guidance_scale,
                    consistency_guidance=None,
                    )
            elif VIS_SPREADS:
                print("Visualization spread step...")
                actions = self.predict_action_chunk(
                    batch,
                    num_samples=10,
                    guidance_actions=None,
                    guidance_scale=None,
                    consistency_guidance=None,
                    )
                _, _ = visualize_trajectories_on_camera(
                    batch,
                    actions.cpu().numpy(),
                    robot,
                    actions_are_normalized=True,
                    postprocessor=postprocessor,
                    name=f"predicted_actions_{self.count}",
                    save_imgs=True,
                    )
                actions = actions[0:1]
            else:  # no intervention
                print(f"No intervention ...")
                actions = self.predict_action_chunk(
                    batch,
                    num_samples=1,
                    guidance_actions=None,
                    guidance_scale=guidance_scale,
                    consistency_guidance=None,
                    )
            # # just to vis actions distribution
            # actions = self.predict_action_chunk(
            #     batch,
            #     num_samples=num_trajs,
            #     guidance_actions=None,
            #     guidance_scale=None,
            #     consistency_guidance=consistency_guidance_action,
            #     )
            # front_prompt_img, wrist_prompt_img = visualize_trajectories_on_camera(
            #     batch,
            #     actions.cpu().numpy(),
            #     robot,
            #     actions_are_normalized=True,
            #     postprocessor=postprocessor,
            #     name=f"predicted_actions_{self.count}",
            #     save_imgs=True,
            #     )
            # actions = actions[0:1]
            self.count += 1
            self._action_queue.extend(actions.transpose(0, 1))
            # sys.exit(0)
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        num_samples: int = 1,
        guidance_actions: torch.FloatTensor = None,
        guidance_scale: float = 1.0,
        gripper_guidance: bool = True,
        consistency_guidance: torch.FloatTensor = None,
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        if consistency_guidance is not None and guidance_actions is not None:
            raise RuntimeError("Currently don't support both consistency guidance and regular guidance")

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # import ipdb;ipdb.set_trace()
        # Sample actions using the model (no separate state needed for PI05)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            num_samples=num_samples,
            guidance_actions=guidance_actions,
            guidance_scale=guidance_scale,
            gripper_guidance=gripper_guidance,
            consistency_guidance=consistency_guidance,
            )
        # print("Normalized actions:", actions.cpu().numpy(), file=sys.stderr)
        # ipdb.set_trace()
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[..., :original_action_dim]

        return actions[:, :self.config.n_action_steps]

    @torch.no_grad()
    def predict_action_chunk_and_get_feature(self, batch: dict[str, Tensor], noise: Tensor = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # import ipdb;ipdb.set_trace()
        # Sample actions using the model (no separate state needed for PI05)
        actions, feature = self.model.sample_actions_and_get_feature(images, img_masks, tokens, masks, noise)

        # ipdb.set_trace()
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, feature

    @torch.no_grad()
    def add_and_denoise1step_get_feature(self, batch: dict[str, Tensor], noise: Tensor) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        clean_action = self.prepare_action(batch)

        # import ipdb;ipdb.set_trace()
        # Sample actions using the model (no separate state needed for PI05)
        actions, feature = self.model.add_and_denoise1step_get_feature(images, img_masks, tokens, masks, clean_action.clone(), noise.clone())

        # ipdb.set_trace()
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, feature


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""
        # import ipdb;ipdb.set_trace()
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss = losses.mean()

        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        return loss, loss_dict

    def action_ensemble(self, action1, action2, batch, postprocessor, robot, action1_weight=0.5, save_imgs=False):
        assert action1.shape[0] == 1
        assert action2.shape[0] == 1
        action2_weight = 1.0 - action1_weight
        combined_action = action1 * action1_weight + action2 * action2_weight
        if save_imgs:
            _, _ = visualize_trajectories_on_camera(
                batch,
                combined_action.cpu().numpy(),
                robot=robot,
                actions_are_normalized=True,
                postprocessor=postprocessor,
                name=f"ensemble_action_{self.count}",
                save_imgs=True,
                save_dir=VLM_IO_OUTPUT_DIR,
                )
        return combined_action

    def primitive_guidance(self, batch, postprocessor, robot, save_imgs=False):
        """Prompt the VLM to select from pre-defined primitives"""
        primitive_labels = ["left", "right", "up", "down", "forward", "backward", "rotate_cw", "rotate_ccw"]
        primitive_actions = torch.cat(
            [
                get_guidance_action_from_text(
                    key,
                    postprocessor=postprocessor,
                    robot=robot,
                )
                for key in primitive_labels
            ],
            dim=0,
        )
        task = batch['task'][0].split(',')[0].split(':')[1].strip()
        text_prompt = self.gen_primitive_text_prompt(task)
        # front_prompt_img, wrist_prompt_img = visualize_trajectories_on_camera(
        #     batch,
        #     primitive_actions.cpu().numpy(),
        #     robot,
        #     actions_are_normalized=True,
        #     postprocessor=postprocessor,
        #     name=f"predicted_actions_{self.count}",
        #     save_imgs=save_imgs,
        #     )
        image_tensor = batch['observation.images.camera_front']
        image_np = (image_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        front_prompt_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        if USE_WRIST:
            raise NotImplementedError
            pil_img = Image.fromarray(cv2.cvtColor(wrist_prompt_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(front_prompt_img, cv2.COLOR_BGR2RGB))
        chosen_label, generated_text = self.vlm_client.select_trajectories(
            pil_img,
            text_prompt,
            len(primitive_labels),
        )
        chosen_label = str(chosen_label).strip().lower()
        if chosen_label == "none":
            chosen_idx = 0
        elif chosen_label in primitive_labels:
            chosen_idx = primitive_labels.index(chosen_label)
        else:
            print(f"Unknown primitive '{chosen_label}', defaulting to '{primitive_labels[0]}'")
            chosen_idx = 0
        print(f"Selected primitive number {chosen_idx} label: {primitive_labels[chosen_idx]}")
        print(f"Reasoning: {generated_text}")
        save_VLM_io(pil_img, generated_text, self.count, prompt_text=text_prompt, suffix="primitive")
        guidance_action = primitive_actions[chosen_idx: chosen_idx + 1]
        return guidance_action.to(batch['observation.state'].device)

    def pivot(self, batch, postprocessor, robot, save_imgs=False):
        num_trajs = 5
        actions = self.predict_action_chunk(
            batch,
            num_samples=num_trajs * 3,
            guidance_actions=None,
            guidance_scale=None,
            consistency_guidance=None,
            )
        if TRAJ_STD_PERTURB > 0.0:
            # Apply perturbation only to trajectory points, not the origin (first point)
            perturbation = torch.randn_like(actions[:, 1:, :3]) * TRAJ_STD_PERTURB
            actions[:, 1:, :3] += perturbation
        idcs = select_representative_trajectories(actions, num_trajectories=num_trajs)
        actions = actions[idcs]
        front_prompt_img, wrist_prompt_img = visualize_trajectories_on_camera(
            batch,
            actions.cpu().numpy(),
            robot,
            actions_are_normalized=True,
            postprocessor=postprocessor,
            name=f"predicted_actions_{self.count}",
            save_imgs=save_imgs,
            )
        task = batch['task'][0].split(',')[0].split(':')[1].strip()
        text_prompt = self.gen_pivot_text_prompt(task, num_trajs)
        if USE_WRIST:
            pil_img = Image.fromarray(cv2.cvtColor(wrist_prompt_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(front_prompt_img, cv2.COLOR_BGR2RGB))
        chosen_color, generated_text = self.vlm_client.select_trajectories(
            pil_img,
            text_prompt,
            num_trajs,
            )
        traj_idx = color2idx(chosen_color)
        print(f"Selected action number {traj_idx} color: {chosen_color}")
        print(f"Reasoning: {generated_text}")
        save_VLM_io(pil_img, generated_text, self.count, prompt_text=text_prompt, suffix="pivot")
        guidance_action = actions[traj_idx: traj_idx + 1]
        return guidance_action


def visualize_trajectories_on_camera(
    batch,
    action,
    robot,
    actions_are_normalized=False,
    postprocessor=None,
    name="test",
    save_imgs=False,
    save_dir=None,
    ):
    padding = 0
    if actions_are_normalized:
        q01 = postprocessor.steps[0].stats['action']['min']
        q99 = postprocessor.steps[0].stats['action']['max']
        denom = q99 - q01
        # chunk = 2.0 * (chunk - q01) / denom - 1.0
        action = (action + 1.0) / 2.0 * denom + q01
    outputs = {}
    for img_key in ['observation.images.camera_front', 'observation.images.camera_wrist']:
        image_tensor = batch[img_key]
        # save the image tensor as an image file
        assert image_tensor.shape[0] == 1 # batch size 1
        image_np = (image_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        image_np = (image_np * 0.80).astype(np.uint8)  # decrease brightness to make trajectories more visible
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np = np.pad(image_np, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
        if "front" in img_key:
            extrinsic, intrinsic = apriltag2cam(padding=padding)
            line_thickness = 1
            draw_arrow_head = False
            downsample = 10
        else: # wrist cam
            extrinsic, intrinsic = get_wrist_cam_mats(robot, padding=padding)
            line_thickness = 2
            draw_arrow_head = True
            downsample = 5
        for i in range(action.shape[0]):
            traj_color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
            points_2D, depths = project_3d_to_2d(action[i, ::downsample, :3], extrinsic, intrinsic)
            draw_lines(points_2D, image_np, depths > 0, traj_color, i==0, line_thickness=line_thickness, draw_arrow_head=draw_arrow_head)
        # Add legend with color names
        legend_y = 30
        for i, color in enumerate(TRAJ_COLORS[:action.shape[0]]):
            cv2.circle(image_np, (30, legend_y), 6, color, -1)
            cv2.putText(image_np, TRAJ_COLOR_NAMES[i], (45, legend_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            legend_y += 25
        outputs[img_key] = image_np
        if save_imgs:
            img_name = f"{name}_{img_key.split('.')[-1]}.png"
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                img_name = os.path.join(save_dir, img_name)
            cv2.imwrite(img_name, image_np)

    return outputs['observation.images.camera_front'], outputs['observation.images.camera_wrist']


def draw_lines(points_2d, img, valid_mask, traj_color, first_traj, line_thickness=1, draw_arrow_head=False):
    last_valid_idx = None
    second_last_valid_idx = None

    for j in range(len(points_2d)):
        if not valid_mask[j]:
            continue

        x, y = int(points_2d[j, 0]), int(points_2d[j, 1])

        # Check if point is within image bounds
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            # if j == 0 and i == 0:  # Only draw current position once
            if j == 0 and first_traj:
                # Current position - larger circle with white outline
                cv2.circle(img, (x, y), 4 * line_thickness, (0, 255, 0), -1)
                cv2.circle(img, (x, y), 5 * line_thickness, (255, 255, 255), 2)
            elif j > 0:
                # Future positions
                cv2.circle(img, (x, y), 2 * line_thickness, traj_color, -1)

            # Track valid points for arrow head
            second_last_valid_idx = last_valid_idx
            last_valid_idx = j

        # Draw line connecting points
        if j > 0 and valid_mask[j-1]:
            x_prev, y_prev = int(points_2d[j-1, 0]), int(points_2d[j-1, 1])
            if (0 <= x_prev < img.shape[1] and 0 <= y_prev < img.shape[0] and
                0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
                cv2.line(img, (x_prev, y_prev), (x, y), traj_color, line_thickness)

    # Draw arrow head at the end of trajectory
    if last_valid_idx is not None and second_last_valid_idx is not None and draw_arrow_head:
        x_last, y_last = int(points_2d[last_valid_idx, 0]), int(points_2d[last_valid_idx, 1])
        x_prev, y_prev = int(points_2d[second_last_valid_idx, 0]), int(points_2d[second_last_valid_idx, 1])

        if (0 <= x_last < img.shape[1] and 0 <= y_last < img.shape[0]):
            # Calculate arrow direction
            dx = x_last - x_prev
            dy = y_last - y_prev
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Normalize direction
                dx /= length
                dy /= length

                # Arrow head parameters
                arrow_length = 10
                arrow_angle = np.pi / 6  # 30 degrees

                # Calculate arrow head points
                arrow_tip = (x_last, y_last)
                arrow_left = (
                    int(x_last - arrow_length * (dx * np.cos(arrow_angle) + dy * np.sin(arrow_angle))),
                    int(y_last - arrow_length * (dy * np.cos(arrow_angle) - dx * np.sin(arrow_angle)))
                )
                arrow_right = (
                    int(x_last - arrow_length * (dx * np.cos(arrow_angle) - dy * np.sin(arrow_angle))),
                    int(y_last - arrow_length * (dy * np.cos(arrow_angle) + dx * np.sin(arrow_angle)))
                )

                # Draw arrow head as a filled triangle
                pts = np.array([arrow_tip, arrow_left, arrow_right], np.int32)
                cv2.fillPoly(img, [pts], traj_color)
                cv2.polylines(img, [pts], True, traj_color, line_thickness)


def get_wrist_cam_mats(robot, padding=0):
    if robot.debug:
        eef_pose = np.array(
            [[ 0.99938,  0.03493, -0.00051,  0.45202],
             [ 0.03491, -0.99916, -0.02116,  0.03674],
             [-0.00125,  0.02113, -0.99978,  0.25416],
             [ 0.     ,  0.     ,  0.     ,  1.     ]],
             )
    else:
        eef_pose = robot.operator.robot_interface.last_eef_pose
    eef_pose[:3, 3] += np.array([-0.055, -0.005, 0.1])
    Rz_neg90 = np.array([
        [ 0.0,  1.0, 0.0],
        [-1.0,  0.0, 0.0],
        [ 0.0,  0.0, 1.0],
    ], dtype=float)
    tilt = np.deg2rad(-18.0)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt),  np.cos(tilt)],
    ])
    extrinsic = eef_pose
    extrinsic[:3, :3] = extrinsic[:3, :3] @ Rz_neg90 @ Rx

    extrinsic = np.linalg.inv(eef_pose)
    f = 456.432495117187 # for 360 x 360 img
    intrinsic = np.array([[f, 0.0, 320.0 + padding],[0.0, f, 180.0 + padding],[0.0, 0.0, 1.0]])
    return extrinsic, intrinsic


def apriltag2cam(padding=0):
    cam2tag =  np.array([
        [
            0.9999778383080971,
            -0.006500179334239067,
            -0.001438944504774469,
            0.2861472831363136
        ],
        [
            0.004474176703044461,
            0.8161981776784388,
            -0.5777547200129354,
            -0.20468220828091388
        ],
        [
            0.004929973173864486,
            0.5777354778988548,
            0.8162091722968363,
            1.5051559145448294
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])
    robot2tag = np.eye(4)

    tag_size = 0.099
    # Move to surface of robot base. aligned with front tip. 3 cm off the ground. 2.3 cm from the side
    tag_translation = np.array([
        0.225 / 2.0 + 0.005,  # add width of clipboard and screws offset
        .508 / 2.0 -0.006 - tag_size / 5 + tag_size / 2.0,
        -0.14 / 2.0 + (tag_size * 9/5 / 2) + 0.043 - tag_size / 5,
        ])
    tag_translation[2] = 0.15  # from trial-and-error based on image results
    tag_translation[0] -= 0.01  # from trial-and-error based on image results
    robot2tag[:3, 3] = tag_translation
    z_correction = np.array([
            [ 0.9990482, -0.0436194,  0.0],
            [ 0.0436194,  0.9990482,  0.0],
            [ 0.0,        0.0,        1.0]
        ]) # from trial-and-error based on image results (2.5 deg about world z)
    # the rotation from robot frame to tag frame is [-90 about y] @ [90 about z (intrinsic)]
    robot2tag[:3, :3] = z_correction @ np.array([[0, 0, -1],
                                [1, 0, 0],
                                [0, -1, 0]])

    # ================================================================================
    # Do the actual transformation to get camera position from tag
    robot2cam = robot2tag @ np.linalg.inv(cam2tag)
    extrinsic = np.linalg.inv(robot2cam)

    # ================================================================================
    # set correct camera instrinsics
    f = 456.432495117187 # for 360 x 360 img
    intrinsic = np.array([[f, 0.0, 320.0 + padding],[0.0, f, 180.0 + padding],[0.0, 0.0, 1.0]])
    return extrinsic, intrinsic


def project_3d_to_2d(points_3d, extrinsic, intrinsic):
    """
    Project 3D points in world coordinates to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        extrinsic: (4, 4) camera extrinsic matrix [R|t]
        intrinsic: (3, 3) camera intrinsic matrix

    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
        depths: (N,) array of depths (for filtering points behind camera)
    """
    # Convert to homogeneous coordinates
    points_3d_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)

    # Transform to camera coordinates
    points_cam = (extrinsic @ points_3d_homo.T).T

    # Extract x, y, z in camera frame
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]

    # Project to image plane using intrinsics
    points_2d_homo = (intrinsic @ points_cam[:, :3].T).T

    # Normalize by depth
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]

    return points_2d, z_cam


def select_representative_trajectories(action_trajectories_orig: np.ndarray, num_trajectories: int = 3) -> np.ndarray:
    """
    Selects representative trajectories from a batch with maximum spread.

    This function uses an efficient O(k*n) Farthest Point Sampling (FPS) implementation.

    Strategy:
    1. Start with the "average" (medoid) trajectory — the one most similar overall.
    2. Iteratively select the trajectory most dissimilar to the selected set,
       using a greedy "farthest-point" criterion based on cosine distance.

    Args:
        action_trajectories (np.ndarray): Array of shape (num_samples, horizon, 3) containing 3D positions.
        num_trajectories (int): Number of representative trajectories to select.

    Returns:
        np.ndarray: Indices of selected trajectories.
    """
    action_trajectories = action_trajectories_orig[:, :, :3].detach().cpu().numpy()
    num_samples = action_trajectories.shape[0]

    # Handle edge cases
    if num_trajectories <= 0:
        return np.array([], dtype=int)
    if num_samples <= num_trajectories:
        return np.arange(num_samples)

    # Flatten each trajectory: (num_samples, horizon * 3)
    flattened = action_trajectories.reshape(num_samples, -1)

    # Normalize to avoid zero-vector issues in cosine similarity
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    flattened = np.where(norms == 0, 0, flattened / np.maximum(norms, 1e-8))

    # Compute cosine similarity and convert to distance
    # sim_matrix[i, j] = similarity between trajectory i and j
    sim_matrix = cosine_similarity(flattened)
    dist_matrix = 1 - sim_matrix  # cosine distance (0 = identical, 2 = opposite)

    # Step 1: Select the medoid (most central trajectory)
    # This is the point with the minimum *total* distance to all other points.
    total_dissimilarity = dist_matrix.sum(axis=1)
    medoid_idx = np.argmin(total_dissimilarity)

    selected_indices = [medoid_idx]

    # --- Optimized Step 2 ---
    # We maintain an array of the minimum distance from each point
    # to the *currently selected set*.
    # Initialize it with distances to the first selected point (the medoid).
    min_dists_to_selected = dist_matrix[medoid_idx, :].copy()

    # Mark the medoid as "selected" by setting its min_dist to a value
    # that np.argmax will never pick (since distances are >= 0).
    min_dists_to_selected[medoid_idx] = -1.0

    for _ in range(num_trajectories - 1):
        # 1. Find the point with the maximum "minimum distance"
        # This is the point farthest from its closest selected neighbor.
        farthest_idx = np.argmax(min_dists_to_selected)

        # This check handles the case where we've selected all valid points
        if min_dists_to_selected[farthest_idx] == -1.0:
            break

        # 2. Add this point to our set
        selected_indices.append(farthest_idx)

        # 3. Mark this new point as selected
        min_dists_to_selected[farthest_idx] = -1.0

        # 4. Update the min_dists array.
        # For all remaining points, their new minimum distance is
        # min(their_current_min_dist, their_dist_to_the_new_point).
        dists_to_new_point = dist_matrix[farthest_idx, :]
        min_dists_to_selected = np.minimum(min_dists_to_selected, dists_to_new_point)

    selected_indices = np.array(selected_indices, dtype=int)
    # cast back to torch gpu tensor
    selected_indices = torch.from_numpy(selected_indices).to(action_trajectories_orig.device)
    return selected_indices


if __name__ == "__main__":
    import pickle
    for i in range(5):
        with open(f"input_args_{i}.pkl", "rb") as f:
            imgs, postprocessor, actions = pickle.load(f)
        visualize_trajectories_on_camera(
            imgs,
            actions,
            postprocessor=postprocessor,
            actions_are_normalized=True,
            name=f"test_{i}_new_fix"
            )
