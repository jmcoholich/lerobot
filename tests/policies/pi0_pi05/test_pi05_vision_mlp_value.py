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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.modeling_vision_mlp_value import _load_vision_weights
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.scripts.lerobot_train import evaluate_scalar_predictor
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


class _FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.vision_model = nn.Module()
        self.vision_model.embeddings = nn.Module()
        self.vision_model.embeddings.num_positions = 4
        self.vision_model.embeddings.patch_embedding = nn.Conv2d(3, 6, kernel_size=1)
        self.vision_model.embeddings.register_buffer(
            "position_ids", torch.arange(4).expand((1, -1)), persistent=False
        )
        self.proj = nn.Linear(3, 6)

    def forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(-1, -2))
        hidden = self.proj(pooled)[:, None].expand(-1, 4, -1)
        return SimpleNamespace(last_hidden_state=hidden)

    def gradient_checkpointing_enable(self):
        pass

    def gradient_checkpointing_disable(self):
        pass


def _make_config(**overrides) -> PI05Config:
    kwargs = {
        "use_value_model": True,
        "value_backbone": "vision_mlp",
        "value_key": "return",
        "value_dim": 1,
        "max_state_dim": 4,
        "vision_mlp_projection_dim": 4,
        "vision_mlp_hidden_dim": 5,
        "vision_mlp_dropout": 0.0,
        "dtype": "float32",
        "device": "cpu",
    }
    kwargs.update(overrides)
    config = PI05Config(**kwargs)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    return config


def _make_policy(monkeypatch, **config_overrides) -> PI05Policy:
    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_vision_mlp_value.SiglipVisionModel",
        lambda config: _FakeVisionModel(),
    )
    return PI05Policy(_make_config(**config_overrides))


def test_vision_mlp_value_policy_has_no_language_model_or_language_inputs(monkeypatch):
    policy = _make_policy(monkeypatch)
    batch = {
        OBS_STATE: torch.randn(2, 4),
        OBS_IMAGE: torch.rand(2, 3, 8, 8),
        "return": torch.ones(2),
    }

    loss, output = policy(batch)
    loss.backward()

    assert loss.shape == ()
    assert output["_iql_predictions"].shape == (2, 1)
    assert policy.model.value_head.weight.grad is not None
    assert not any("language" in name or "token" in name for name, _ in policy.model.named_parameters())


@pytest.mark.parametrize(("value_loss", "expected_loss"), [("l1", 0.5), ("l2", 0.25)])
def test_vision_mlp_value_policy_supports_l1_and_l2_loss(monkeypatch, value_loss, expected_loss):
    policy = _make_policy(monkeypatch, value_loss=value_loss)
    batch = {
        OBS_STATE: torch.randn(2, 4),
        OBS_IMAGE: torch.rand(2, 3, 8, 8),
        "return": torch.full((2,), 0.5),
    }

    loss, _ = policy(batch)

    assert loss.item() == pytest.approx(expected_loss)


def test_scalar_evaluation_reports_separate_l1_and_l2_losses(monkeypatch):
    policy = _make_policy(monkeypatch)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    batch = {
        OBS_STATE: torch.zeros(2, 4),
        OBS_IMAGE: torch.zeros(2, 3, 8, 8),
        "return": torch.tensor([0.5, 2.0]),
    }

    class _Accelerator:
        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def gather_for_metrics(values):
            return values

    metrics = evaluate_scalar_predictor(
        policy,
        [batch],
        lambda value: value,
        _Accelerator(),
        {},
    )

    assert metrics["l1_loss"] == pytest.approx(1.25)
    assert metrics["l2_loss"] == pytest.approx(2.125)
    assert "loss" not in metrics


def test_vision_mlp_uses_one_canonical_image(monkeypatch):
    policy = _make_policy(monkeypatch)

    assert list(policy.config.image_features) == [OBS_IMAGE]
    assert policy.model.num_image_features == 1


def test_vision_mlp_passes_continuous_state(monkeypatch):
    policy = _make_policy(monkeypatch)
    policy.eval()
    seen_features = []
    hook = policy.model.value_mlp[0].register_forward_pre_hook(
        lambda module, args: seen_features.append(args[0].detach().clone())
    )
    state = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    policy.model.predict_values(
        [torch.rand(1, 3, 8, 8)],
        [torch.ones(1, dtype=torch.bool)],
        state,
    )
    hook.remove()

    torch.testing.assert_close(seen_features[0][:, -policy.config.max_state_dim :], state)


def test_vision_mlp_processor_does_not_construct_a_tokenizer(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("vision_mlp must not construct a tokenizer")

    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        fail_if_called,
    )
    preprocessor, _ = make_pi05_pre_post_processors(_make_config())

    assert [type(step).__name__ for step in preprocessor.steps] == [
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "NormalizerProcessorStep",
        "DeviceProcessorStep",
    ]


def test_vision_mlp_can_freeze_only_the_vision_encoder(monkeypatch):
    policy = _make_policy(monkeypatch, freeze_vision_encoder=True)
    policy.train()

    assert not policy.model.vision_tower.training
    assert not any(parameter.requires_grad for parameter in policy.model.vision_tower.parameters())
    assert all(parameter.requires_grad for parameter in policy.model.value_head.parameters())


@pytest.mark.parametrize(
    "prefix",
    [
        "vision_tower.",
        "paligemma_with_expert.paligemma.model.vision_tower.",
    ],
)
def test_selective_loader_accepts_paligemma_and_pi05_prefixes(tmp_path, prefix):
    source = _FakeVisionModel()
    expected = {key: value.detach().clone() for key, value in source.state_dict().items()}
    checkpoint = {f"{prefix}{key}": value for key, value in expected.items()}
    checkpoint["language_model.layers.0.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint, checkpoint_path)
    with torch.device("meta"):
        target = _FakeVisionModel()

    _load_vision_weights(target, str(checkpoint_path))

    assert all(buffer.device.type != "meta" for buffer in target.buffers())
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_vision_mlp_checkpoint_round_trip(monkeypatch, tmp_path):
    policy = _make_policy(monkeypatch)
    with torch.no_grad():
        policy.model.value_head.bias.fill_(3.0)
    policy.save_pretrained(tmp_path)

    loaded = PI05Policy.from_pretrained(tmp_path, config=policy.config)

    assert list(loaded.config.image_features) == [OBS_IMAGE]
    assert loaded.model.num_image_features == 1
    torch.testing.assert_close(loaded.model.value_head.bias, policy.model.value_head.bias)


def test_vision_mlp_config_validation():
    with pytest.raises(ValueError, match="vision_mlp_projection_dim"):
        _make_config(vision_mlp_projection_dim=0)
    with pytest.raises(ValueError, match="vision_mlp_hidden_dim"):
        _make_config(vision_mlp_hidden_dim=0)
    with pytest.raises(ValueError, match="vision_mlp_dropout"):
        _make_config(vision_mlp_dropout=1.0)
    with pytest.raises(ValueError, match="value_loss"):
        _make_config(value_loss="huber")


def test_vision_mlp_requires_square_images(monkeypatch):
    with pytest.raises(ValueError, match="square image resolution"):
        _make_policy(monkeypatch, image_resolution=(8, 4))
