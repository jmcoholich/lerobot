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
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

IMAGE_KEY = "observation.images.test"


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


def _make_config(**overrides) -> PI05Config:
    kwargs = {
        "use_value_model": True,
        "value_backbone": "smolvlm",
        "value_key": "return",
        "value_dim": 1,
        "max_state_dim": 4,
        "dtype": "float32",
        "device": "cpu",
    }
    kwargs.update(overrides)
    config = PI05Config(
        **kwargs,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    return config


def _make_policy(monkeypatch, **config_overrides) -> PI05Policy:
    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_smolvlm_value.AutoModelForImageTextToText.from_pretrained",
        lambda *args, **kwargs: _FakeSmolVLM(),
    )
    return PI05Policy(_make_config(**config_overrides))


def _make_bootstrap_batch(
    rewards=(1.0, 2.0, 3.0),
    *,
    reward_is_pad=None,
    future_is_pad=False,
    mc_target=9.0,
):
    if reward_is_pad is None:
        reward_is_pad = [False] * len(rewards)
    return {
        OBS_STATE: torch.tensor([[[1.0] * 4, [9.0] * 4]]),
        IMAGE_KEY: torch.stack(
            [torch.zeros(3, 8, 8), torch.ones(3, 8, 8)],
        ).unsqueeze(0),
        "sparse_reward": torch.tensor(rewards).reshape(1, -1, 1),
        "sparse_reward_is_pad": torch.tensor([reward_is_pad]),
        f"{OBS_STATE}_is_pad": torch.tensor([[False, future_is_pad]]),
        f"{IMAGE_KEY}_is_pad": torch.tensor([[False, future_is_pad]]),
        "return": torch.tensor([[mc_target]]),
    }


def test_smolvlm_value_policy_ignores_language(monkeypatch):
    policy = _make_policy(monkeypatch)
    batch = {
        OBS_STATE: torch.randn(2, 4),
        IMAGE_KEY: torch.rand(2, 3, 8, 8),
        "return": torch.ones(2),
    }

    loss, _ = policy(batch)

    assert loss.shape == ()
    loss.backward()
    assert policy.model.value_head.weight.grad is not None
    assert policy.predict_values(batch).shape == (2, 1)


def test_smolvlm_bootstrap_target_is_frozen_eval_and_no_grad(monkeypatch):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_discount=0.5,
        value_reward_key="sparse_reward",
        value_target_tau=0.25,
    )
    grad_enabled = []
    predict_values = policy.model_target.predict_values

    def record_grad_mode(*args, **kwargs):
        grad_enabled.append(torch.is_grad_enabled())
        return predict_values(*args, **kwargs)

    monkeypatch.setattr(policy.model_target, "predict_values", record_grad_mode)
    policy.train()

    loss, _ = policy(_make_bootstrap_batch())
    loss.backward()

    assert not policy.model_target.training
    assert all(not parameter.requires_grad for parameter in policy.model_target.parameters())
    assert grad_enabled == [False]
    assert all(parameter.grad is None for parameter in policy.model_target.parameters())
    assert policy.model.value_head.weight.grad is not None


def test_smolvlm_bootstrap_builds_exact_interior_n_step_target(monkeypatch):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_discount=0.5,
        value_reward_key="sparse_reward",
    )
    monkeypatch.setattr(
        policy.model,
        "predict_values",
        lambda images, img_masks, state: torch.zeros((state.shape[0], 1)),
    )
    monkeypatch.setattr(
        policy.model_target,
        "predict_values",
        lambda images, img_masks, state: torch.full((state.shape[0], 1), 8.0),
    )
    policy.train()

    loss, output = policy(_make_bootstrap_batch(rewards=(1.0, 2.0, 3.0)))

    expected_target = torch.tensor([[3.75]])  # 1 + .5*2 + .5^2*3 + .5^3*8
    torch.testing.assert_close(output["_iql_targets"], expected_target)
    torch.testing.assert_close(loss, expected_target.square().mean())


def test_smolvlm_bootstrap_masks_padded_rewards_and_terminal_bootstrap(monkeypatch):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_discount=0.5,
        value_reward_key="sparse_reward",
    )
    monkeypatch.setattr(
        policy.model,
        "predict_values",
        lambda images, img_masks, state: torch.zeros((state.shape[0], 1)),
    )
    monkeypatch.setattr(
        policy.model_target,
        "predict_values",
        lambda images, img_masks, state: torch.full((state.shape[0], 1), 100.0),
    )
    policy.train()

    loss, output = policy(
        _make_bootstrap_batch(
            rewards=(1.0, 1.0, 1.0),
            reward_is_pad=(False, True, True),
            future_is_pad=True,
        )
    )

    expected_target = torch.tensor([[1.0]])
    torch.testing.assert_close(output["_iql_targets"], expected_target)
    torch.testing.assert_close(loss, expected_target.square().mean())


def test_smolvlm_bootstrap_uses_current_and_future_observation_slots(monkeypatch):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_discount=0.5,
        value_reward_key="sparse_reward",
    )
    seen = {}

    def predict_current(images, img_masks, state):
        seen["current_state"] = state.detach().clone()
        seen["current_image"] = images[0].detach().clone()
        return torch.zeros((state.shape[0], 1))

    def predict_future(images, img_masks, state):
        seen["future_state"] = state.detach().clone()
        seen["future_image"] = images[0].detach().clone()
        return torch.zeros((state.shape[0], 1))

    monkeypatch.setattr(policy.model, "predict_values", predict_current)
    monkeypatch.setattr(policy.model_target, "predict_values", predict_future)
    policy.train()

    policy(_make_bootstrap_batch())

    torch.testing.assert_close(seen["current_state"], torch.ones(1, 4))
    torch.testing.assert_close(seen["future_state"], torch.full((1, 4), 9.0))
    torch.testing.assert_close(seen["current_image"], -torch.ones(1, 3, 8, 8))
    torch.testing.assert_close(seen["future_image"], torch.ones(1, 3, 8, 8))


def test_smolvlm_bootstrap_polyak_update_uses_tau(monkeypatch):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_target_tau=0.25,
    )
    with torch.no_grad():
        policy.model.value_head.weight.fill_(4.0)
        policy.model_target.value_head.weight.zero_()

    policy.update()

    torch.testing.assert_close(
        policy.model_target.value_head.weight,
        torch.ones_like(policy.model_target.value_head.weight),
    )


def test_smolvlm_bootstrap_eval_uses_dataset_mc_target(monkeypatch):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_discount=0.5,
        value_reward_key="sparse_reward",
    )
    monkeypatch.setattr(
        policy.model,
        "predict_values",
        lambda images, img_masks, state: torch.full((state.shape[0], 1), 2.0),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("The target model must not be used for MC evaluation")

    monkeypatch.setattr(policy.model_target, "predict_values", fail_if_called)
    policy.eval()

    loss, output = policy(_make_bootstrap_batch(mc_target=5.0))

    torch.testing.assert_close(output["_iql_targets"], torch.tensor([[5.0]]))
    torch.testing.assert_close(loss, torch.tensor(9.0))


def test_smolvlm_value_processor_omits_tokenizer():
    preprocessor, _ = make_pi05_pre_post_processors(_make_config())
    step_names = [type(step).__name__ for step in preprocessor.steps]

    assert step_names == [
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "NormalizerProcessorStep",
        "DeviceProcessorStep",
    ]


def test_smolvlm_bootstrap_requests_future_observations_and_sparse_rewards():
    config = _make_config(value_bootstrap_steps=3)
    metadata = SimpleNamespace(
        fps=20,
        features={
            ACTION: None,
            OBS_STATE: None,
            IMAGE_KEY: None,
            "sparse_reward": None,
            "return": None,
        },
    )

    delta_timestamps = resolve_delta_timestamps(config, metadata)

    assert delta_timestamps["sparse_reward"] == [0.0, 0.05, 0.1]
    assert delta_timestamps[OBS_STATE] == [0.0, 0.15]
    assert delta_timestamps[IMAGE_KEY] == [0.0, 0.15]


def test_smolvlm_bootstrap_processor_keeps_targets_in_raw_units():
    preprocessor, _ = make_pi05_pre_post_processors(
        _make_config(value_bootstrap_steps=3),
    )

    assert preprocessor.steps[2].normalize_complementary_data_keys is None


def test_smolvlm_value_checkpoint_round_trip(monkeypatch, tmp_path):
    policy = _make_policy(
        monkeypatch,
        value_bootstrap_steps=3,
        value_reward_key="sparse_reward",
    )
    with torch.no_grad():
        policy.model.value_head.weight.fill_(3.0)
        policy.model_target.value_head.weight.fill_(2.0)
    policy.save_pretrained(tmp_path)

    loaded = PI05Policy.from_pretrained(tmp_path, config=policy.config)

    assert isinstance(loaded.model.vlm, _FakeSmolVLM)
    torch.testing.assert_close(loaded.model.value_head.weight, policy.model.value_head.weight)
    torch.testing.assert_close(
        loaded.model_target.value_head.weight,
        policy.model_target.value_head.weight,
    )


def test_smolvlm_bootstrap_syncs_target_when_loading_legacy_checkpoint(monkeypatch, tmp_path):
    legacy_policy = _make_policy(monkeypatch)
    with torch.no_grad():
        legacy_policy.model.value_head.weight.fill_(3.0)
    legacy_policy.save_pretrained(tmp_path)

    bootstrap_config = _make_config(
        value_bootstrap_steps=3,
        value_reward_key="sparse_reward",
    )
    loaded = PI05Policy.from_pretrained(tmp_path, config=bootstrap_config)

    torch.testing.assert_close(loaded.model_target.value_head.weight, loaded.model.value_head.weight)
