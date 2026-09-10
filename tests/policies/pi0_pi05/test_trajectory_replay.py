"""Replay real policy orchestration against a deterministic CPU sampler stub."""

import ast
import copy
from collections import deque
from dataclasses import asdict, make_dataclass
import hashlib
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


POLICY_DIR = Path(__file__).resolve().parents[3] / "src/lerobot/policies/pi05"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, POLICY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder_module = load_module("trajectory_recorder")
replay_module = load_module("trajectory_replay")


class Sampler:
    def eval(self):
        return self

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def sample_actions(self, images, image_masks, tokens, masks, *, noise, num_samples,
                       guidance_actions=None, guidance_scale=None, gripper_guidance=True,
                       consistency_guidance=None, num_steps=10):
        # Every recorded input contributes to the result; changing the RNG must not.
        value = sum(image.mean() * mask.float().mean() for image, mask in zip(images, image_masks, strict=True))
        value += (tokens * masks).float().mean() / 1000
        output = noise + value / num_steps
        if guidance_actions is not None:
            output += guidance_actions.mean() * guidance_scale
        return output


class TestTrajectoryReplay(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.namespace = dict(
            torch=torch, Tensor=torch.Tensor, np=np, json=json, sys=sys,
            as_numpy=recorder_module.as_numpy, cosine_similarity=cosine_similarity,
            INTERVENTIONS=False, MANUAL_GUIDANCE=False, VIS_SPREADS=False,
            TRAJ_STD_PERTURB=0.01, MAX_CHUNKS=8, USE_WRIST=False,
            OBS_LANGUAGE_TOKENS="observation.language.tokens",
            OBS_LANGUAGE_ATTENTION_MASK="observation.language.attention_mask", ACTION="action",
            copy=copy, asdict=asdict, hashlib=hashlib, version=version, os=os, Path=Path,
            __file__=str(POLICY_DIR / "modelling_pi05_taco.py"),
            TrajectoryRecorder=recorder_module.TrajectoryRecorder,
            snapshot_processor=recorder_module.snapshot_processor,
        )
        # Avoid model/hardware imports, but run the actual policy methods being changed.
        tree = ast.parse((POLICY_DIR / "modelling_pi05_taco.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI05PolicyTaco")
        names = ("configure_replay_recording", "_ensure_trajectory_recorder", "capture_replay_observation",
                 "_sample_trajectory_candidates", "predict_action_chunk", "select_action")
        nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_representative_trajectories"]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "policy_methods", "exec"), self.namespace)
        stub_cls = type("Policy", (), {name: self.namespace[name] for name in names})
        self.policy = stub_cls()
        self.policy.config = SimpleNamespace(chunk_size=100, max_action_dim=32, n_action_steps=50,
                                             num_inference_steps=10, output_features={"action": SimpleNamespace(shape=(8,))}, device="cpu")
        self.policy.model = Sampler()
        self.policy.eval = Mock()
        self.policy._ensure_trajectory_recorder = Mock()
        self.policy._trajectory_recorder = recorder_module.TrajectoryRecorder({}, self.directory.name, demo_name="replay")
        self.policy._action_queue = deque(maxlen=50)
        self.policy.count = 0
        self.policy._replay_sampling_calls = []
        self.batch = {
            "observation.state": torch.arange(8, dtype=torch.float32)[None] / 10,
            "observation.language.tokens": torch.tensor([[5, 42, 123456]], dtype=torch.int64),
            "observation.language.attention_mask": torch.tensor([[True, True, False]]),
            "task": ["Task: place blocks, State: ..."],
            **{f"observation.images.camera_{name}": torch.rand(1, 3, 8, 8) for name in ("front", "wrist", "side")},
        }
        self.policy._preprocess_images = lambda batch: (
            [batch[f"observation.images.camera_{name}"] * 2 - 1 for name in ("front", "wrist", "side")],
            [torch.tensor([True])] * 3,
        )

    def run_replay(self, guided=False):
        if guided:
            self.namespace["INTERVENTIONS"] = "PIVOT"
            self.policy.pivot = lambda *args, actions, **kwargs: actions[:1]
        raw = {"observation.state": torch.arange(8, dtype=torch.float32)[None], "task": ["place blocks"]}
        self.policy.capture_replay_observation(raw, Mock(), Mock())
        expected_state = raw["observation.state"].clone()
        raw["observation.state"].zero_()  # Captured input must not alias mutable preprocessing inputs.
        executed = torch.stack([self.policy.select_action(self.batch) for _ in range(50)], dim=1)
        path = self.policy._trajectory_recorder.path
        with h5py.File(path, "r") as file:
            chunk = file["chunks/000000"]
            self.assertEqual(chunk["sampling/000000/inputs/noise"].shape, (15, 100, 32))
            self.assertEqual(chunk["sampling/000000/inputs/tokens"].dtype, np.dtype("int64"))
            np.testing.assert_array_equal(chunk["raw_observation/observation.state"], expected_state.numpy())
            self.assertEqual(len(chunk["sampling"]), 2 if guided else 1)
            if guided:
                self.assertIn("guidance_actions", chunk["sampling/000001/inputs"])
        torch.manual_seed(987654)
        self.policy.model.sample_noise = Mock(side_effect=AssertionError("Replay must use recorded noise"))
        replayed = replay_module.replay_trajectory_chunk(self.policy, path)
        torch.testing.assert_close(replayed, executed, rtol=0, atol=0)

    def test_unguided_replay_with_different_rng(self):
        self.run_replay()

    def test_guided_replay_captures_second_sampling_call(self):
        self.run_replay(guided=True)

    def test_explicit_noise_bypasses_rng(self):
        noise = torch.ones(15, 100, 32)
        self.policy.model.sample_noise = Mock(side_effect=AssertionError("Unexpected random draw"))
        self.policy.predict_action_chunk(self.batch, num_samples=15, noise=noise)
        np.testing.assert_array_equal(self.policy._replay_sampling_calls[0]["inputs"]["noise"], noise.numpy())

    def test_full_rollout_configuration_and_launch_metadata_are_saved(self):
        self.policy._trajectory_recorder = None
        self.policy._checkpoint_file = None
        self.policy.config = make_dataclass("Config", [("n_action_steps", int)])(50)
        del self.policy._ensure_trajectory_recorder  # Use the real initializer for this test.
        config = {
            "dataset": {"single_task": "place both blocks in the bin", "fps": 30, "num_episodes": 1},
            "robot": {"record": "metadata", "port": "dummy"},
            "policy": {"n_action_steps": 50},
        }
        self.policy.configure_replay_recording(config)
        config["dataset"]["single_task"] = "changed after capture"
        argv = ["lerobot_record.py", "--policy.n_action_steps=50", "--robot.record=metadata"]
        with patch.dict(os.environ, {"LEROBOT_TRAJECTORY_DIR": self.directory.name, "LEROBOT_DEMO_NAME": "metadata"}), patch.object(sys, "argv", argv):
            self.policy._ensure_trajectory_recorder()
        with h5py.File(self.policy._trajectory_recorder.path, "r") as file:
            metadata = json.loads(file.attrs["metadata_json"])
            self.assertEqual(metadata["rollout_config"]["dataset"]["single_task"], "place both blocks in the bin")
            self.assertEqual(metadata["rollout_config"]["dataset"]["fps"], 30)
            self.assertEqual(metadata["policy_config"]["n_action_steps"], 50)
            self.assertEqual(metadata["launch"]["argv"], argv)
            self.assertEqual(metadata["launch"]["python_executable"], sys.executable)
            self.assertEqual(metadata["taco_settings"]["max_chunks"], 8)
            for filename in ("collect_eval.bash", "pi_05_inference.bash", "lerobot_record.py", "franka.py"):
                self.assertIn(filename, file["artifacts/source"])
        with self.assertRaises(RuntimeError):
            self.policy.configure_replay_recording(config)


if __name__ == "__main__":
    unittest.main()
