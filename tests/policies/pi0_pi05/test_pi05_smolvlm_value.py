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

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


class _FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 6)

    def forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(-1, -2))
        hidden = self.proj(pooled)[:, None].expand(-1, 4, -1)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, attention_mask, use_cache, return_dict):
        del attention_mask, use_cache, return_dict
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.proj(inputs_embeds))


class _FakeSmolVLMCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _FakeVisionModel()
        self.connector = nn.Linear(6, 8)
        self.text_model = _FakeTextModel()


class _FakeSmolVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=8),
            vision_config=SimpleNamespace(image_size=8),
        )
        self.model = _FakeSmolVLMCore()
        self.lm_head = nn.Linear(8, 32)

    def gradient_checkpointing_enable(self):
        pass

    def gradient_checkpointing_disable(self):
        pass


def _make_config() -> PI05Config:
    config = PI05Config(
        use_value_model=True,
        value_backbone="smolvlm",
        value_key="return",
        value_dim=1,
        max_state_dim=4,
        dtype="float32",
        device="cpu",
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "observation.images.test": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    return config


def test_smolvlm_value_policy_ignores_language(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_smolvlm_value.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: _FakeSmolVLM(),
    )
    policy = PI05Policy(_make_config())
    batch = {
        OBS_STATE: torch.randn(2, 4),
        "observation.images.test": torch.rand(2, 3, 8, 8),
        "return": torch.ones(2),
    }

    loss, _ = policy(batch)

    assert loss.shape == ()
    loss.backward()
    assert policy.model.value_head.weight.grad is not None
    assert policy.predict_values(batch).shape == (2, 1)


def test_smolvlm_value_processor_omits_tokenizer():
    preprocessor, _ = make_pi05_pre_post_processors(_make_config())
    step_names = [type(step).__name__ for step in preprocessor.steps]

    assert step_names == [
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "NormalizerProcessorStep",
        "DeviceProcessorStep",
    ]


def test_smolvlm_value_checkpoint_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_smolvlm_value.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: _FakeSmolVLM(),
    )
    policy = PI05Policy(_make_config())
    policy.save_pretrained(tmp_path)

    loaded = PI05Policy.from_pretrained(tmp_path, config=policy.config)

    assert isinstance(loaded.model.vlm, _FakeSmolVLM)
