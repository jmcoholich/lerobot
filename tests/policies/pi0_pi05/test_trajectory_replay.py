"""Replay real policy orchestration against a deterministic CPU sampler stub."""

import ast
import copy
from collections import deque
from dataclasses import asdict, make_dataclass
import hashlib
from importlib.metadata import version
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import cv2
import numpy as np
from PIL import Image
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
            TRAJ_STD_PERTURB=0.01, MAX_STEPS=800, USE_WRIST=False,
            OBS_LANGUAGE_TOKENS="observation.language.tokens",
            OBS_LANGUAGE_ATTENTION_MASK="observation.language.attention_mask", ACTION="action",
            copy=copy, asdict=asdict, hashlib=hashlib, logging=logging, time=time, version=version, os=os, Path=Path,
            __file__=str(POLICY_DIR / "modelling_pi05_taco.py"),
            TrajectoryRecorder=recorder_module.TrajectoryRecorder,
            snapshot_processor=recorder_module.snapshot_processor,
            VLMClient=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            deque=deque,
        )
        # Avoid model/hardware imports, but run the actual policy methods being changed.
        tree = ast.parse((POLICY_DIR / "modelling_pi05_taco.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI05PolicyTaco")
        names = ("configure_interventions", "reset_rollout", "configure_replay_recording", "_ensure_trajectory_recorder", "capture_replay_observation",
                 "_sample_trajectory_candidates", "predict_action_chunk", "select_action", "reset",
                 "pivot", "primitive_guidance", "action_ensemble")
        nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_representative_trajectories"]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "policy_methods", "exec"), self.namespace)
        stub_cls = type("Policy", (), {name: self.namespace[name] for name in names})
        self.policy = stub_cls()
        self.policy.vlm_client = None
        self.policy.configure_interventions()
        self.policy.config = SimpleNamespace(chunk_size=100, max_action_dim=32, n_action_steps=50,
                                             num_inference_steps=10, output_features={"action": SimpleNamespace(shape=(8,))}, device="cpu")
        self.policy.model = Sampler()
        self.policy.eval = Mock()
        self.policy._ensure_trajectory_recorder = Mock()
        self.policy._trajectory_recorder = recorder_module.TrajectoryRecorder({}, self.directory.name, demo_name="replay")
        self.policy._action_queue = deque(maxlen=50)
        self.policy.count = 0
        self.policy._checkpoint_sha256 = None
        self.policy._replay_sampling_calls = []
        self.policy._raw_replay_observation = {}
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
            self.policy.configure_interventions(interventions="PIVOT", vlm_server_url="http://127.0.0.1:35959")
            def pivot(*args, actions, **kwargs):
                self.policy._intervention_details["pivot_color"] = "Blue"
                return actions[:1]
            self.policy.pivot = pivot
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
            timings = chunk["timing"]
            self.assertGreater(timings["base_inference_s"][()], 0.)
            self.assertGreater(timings["policy_generation_s"][()], timings["base_inference_s"][()])
            if guided:
                self.assertGreater(timings["intervention_s"][()], 0.)
                self.assertAlmostEqual(timings["intervention_s"][()],
                                       timings["guidance_selection_s"][()] + timings["guided_inference_s"][()])
            else:
                self.assertEqual(timings["intervention_s"][()], 0.)
            self.assertEqual(chunk["intervention/occurred"][()], guided)
            self.assertEqual(chunk["intervention"].attrs["setting"], "PIVOT" if guided else "none")
            self.assertTrue(chunk["intervention/triggered"][()])
            if guided:
                self.assertEqual(chunk["intervention"].attrs["pivot_color"], "Blue")
                self.assertIn("guidance_actions", chunk["sampling/000001/inputs"])
        torch.manual_seed(987654)
        self.policy.model.sample_noise = Mock(side_effect=AssertionError("Replay must use recorded noise"))
        replayed = replay_module.replay_trajectory_chunk(self.policy, path)
        torch.testing.assert_close(replayed, executed, rtol=0, atol=0)

    def test_unguided_replay_with_different_rng(self):
        self.run_replay()

    def test_full_guidance_and_execution_horizons_for_all_strategies(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        render = Mock(return_value=(image, None))
        self.namespace.update(
            cv2=cv2, Image=Image, TRAJ_COLOR_NAMES=["Orange", "Blue"], color2idx=lambda color: 1,
            visualize_trajectories_on_camera=render, get_vlm_output_dir=lambda: self.directory.name,
            save_VLM_io=Mock(), get_guidance_action_from_text=Mock(return_value=torch.ones(1, 100, 8)),
        )
        self.policy.gen_pivot_text_prompt = Mock(return_value="")
        self.policy.gen_primitive_text_prompt = Mock(return_value="")
        for steps in (25, 50, 100):
            for strategy in ("PIVOT", "primitive", "ensemble"):
                with self.subTest(steps=steps, strategy=strategy):
                    self.policy.config.n_action_steps = steps
                    self.policy.reset()
                    self.policy.count = 0
                    self.policy.configure_interventions(interventions=strategy, vlm_server_url="http://test")
                    self.policy.vlm_client.select_trajectories = lambda image, prompt, count: (
                        "up" if count == 7 else "blue", "reasoning")
                    recorder = recorder_module.TrajectoryRecorder({}, self.directory.name, demo_name="full_guidance")
                    self.policy._trajectory_recorder = recorder
                    render.reset_mock()
                    with patch.object(recorder_module, "compute_diversity_rbf", return_value=(0.5, 0.063)), \
                         patch("builtins.print"):
                        executed = torch.stack([self.policy.select_action(self.batch) for _ in range(steps)], dim=1)
                    self.assertEqual(executed.shape, (1, steps, 8))
                    self.assertEqual(len(self.policy._action_queue), 0)
                    with h5py.File(recorder.path, "r") as file:
                        chunk = file["chunks/000000"]
                        self.assertEqual(chunk["sampling/000001/inputs/guidance_actions"].shape, (1, 100, 8))
                        self.assertEqual(chunk["sampling/000001/full_output"].shape, (1, 100, 32))
                        self.assertEqual(chunk["queued_actions"].shape, (1, steps, 8))
                        self.assertEqual(chunk["temporal_mmd/overlap_steps"][()], 100 - steps)
                        self.assertTrue(chunk["intervention/occurred"][()])
                    for call in render.call_args_list:
                        self.assertEqual(call.args[1].shape[1], 100)
                    self.assertNotIn("down", [call.args[0] for call in self.namespace["get_guidance_action_from_text"].call_args_list])
                    replayed = replay_module.replay_trajectory_chunk(self.policy, recorder.path)
                    torch.testing.assert_close(replayed, executed, rtol=0, atol=0)

    def test_score_triggers_later_chunks_instead_of_first_two(self):
        for strategy in ("PIVOT", "primitive", "ensemble", "none"):
            with self.subTest(strategy=strategy):
                self.policy.reset()
                self.policy.count = 0
                self.policy.configure_interventions(interventions=strategy, vlm_server_url="http://127.0.0.1:35959")
                recorder = recorder_module.TrajectoryRecorder({}, self.directory.name, demo_name="trigger")
                self.policy._trajectory_recorder = recorder
                self.policy.pivot = Mock(return_value=torch.ones(1, 50, 8))
                self.policy.primitive_guidance = Mock(return_value=torch.ones(1, 50, 8))
                self.policy.action_ensemble = Mock(return_value=torch.ones(1, 50, 8))
                # First chunk: no MMD, below threshold. Second: exactly at threshold.
                # Third: above threshold. Fourth: below, so the trigger must clear.
                with patch.object(recorder_module, "compute_diversity_rbf", return_value=(0.2, 0.063)), \
                     patch.object(recorder_module, "compute_mmd_rbf", side_effect=[(0.2082, 1.), (0.4, 1.), (0.1, 1.)]):
                    for _ in range(200):
                        self.policy.select_action(self.batch)
                with h5py.File(recorder.path, "r") as file:
                    for index in range(4):
                        chunk = file[f"chunks/{index:06d}"]
                        metrics = chunk["intervention"]
                        occurred = index == 2 and strategy != "none"
                        self.assertEqual(metrics["triggered"][()], index == 2)
                        self.assertEqual(metrics["occurred"][()], occurred)
                        self.assertEqual(metrics.attrs["setting"], strategy)
                        self.assertEqual(metrics["threshold"][()], 1.2082)
                        self.assertEqual(metrics["diversity_weight"][()], 5.)
                        self.assertEqual(len(chunk["sampling"]), 2 if occurred else 1)
                    self.assertEqual(file["chunks/000000/intervention/score"][()], 1.)
                    self.assertEqual(file["chunks/000001/intervention/score"][()], 1.2082)
                recorder.intervention_triggered = True
                recorder.reset()
                self.assertFalse(recorder.intervention_triggered)

    def test_rollout_stops_after_at_most_800_actions(self):
        for horizon, expected_chunks in ((100, 8), (50, 16), (25, 32), (30, 26)):
            with self.subTest(horizon=horizon), patch("builtins.print"):
                self.policy.config.n_action_steps = horizon
                self.policy.reset()
                self.policy.count = 0
                self.policy.manual_guidance = False
                self.policy._trajectory_recorder = Mock()
                self.policy._active_trajectory_index = 0
                actions = torch.zeros(1, horizon, 8)
                self.policy._sample_trajectory_candidates = Mock(return_value=(actions, actions))
                for _ in range(expected_chunks * horizon):
                    self.policy.select_action(self.batch)
                for manual_guidance in (False, True):
                    self.policy.manual_guidance = manual_guidance
                    with self.assertRaises(SystemExit) as stopped:
                        self.policy.select_action(self.batch)
                    self.assertEqual(stopped.exception.code, 0)
                self.assertEqual(self.policy._sample_trajectory_candidates.call_count, expected_chunks)
                self.assertEqual(self.policy._trajectory_recorder.record_execution.call_count, expected_chunks)
                self.assertEqual(len(self.policy._action_queue), 0)

    def test_mmd_is_saved_once_per_chunk_and_policy_reset_clears_overlap(self):
        with patch.object(self.policy.model, "sample_actions", wraps=self.policy.model.sample_actions) as sample:
            for _ in range(100):
                self.policy.select_action(self.batch)
            self.assertEqual(sample.call_count, 2)
        with h5py.File(self.policy._trajectory_recorder.path, "r") as file:
            self.assertEqual(len(file["chunks"]), 2)
            previous = file["chunks/000000/sampling/000000/full_output"][:, 50:, :7].reshape(15, -1)
            current = file["chunks/000001/sampling/000000/full_output"][:, :50, :7].reshape(15, -1)
            expected, gamma = recorder_module.compute_mmd_rbf(current, previous)
            fixed_gamma = 0.06302211495246096
            diversity, _ = recorder_module.compute_diversity_rbf(current, fixed_gamma)
            self.assertAlmostEqual(file["chunks/000001/temporal_mmd/mmd2"][()], expected)
            self.assertEqual(file["chunks/000001/temporal_mmd/gamma"][()], gamma)
            self.assertEqual(file["chunks/000001/candidate_diversity/score"][()], diversity)
            self.assertEqual(file["chunks/000001/candidate_diversity/gamma"][()], fixed_gamma)
            self.assertEqual(file["chunks/000001/candidate_diversity/horizon"][()], 50)
            self.assertEqual(file["chunks/000001/candidate_diversity/action_dim"][()], 7)
            first = file["chunks/000000/sampling/000000/full_output"][:, :50, :7].reshape(15, -1)
            first_diversity, first_gamma = recorder_module.compute_diversity_rbf(first, fixed_gamma)
            self.assertEqual(file["chunks/000000/candidate_diversity/score"][()], first_diversity)
            self.assertEqual(file["chunks/000000/candidate_diversity/gamma"][()], first_gamma)
        self.policy.reset()
        self.policy.select_action(self.batch)
        with h5py.File(self.policy._trajectory_recorder.path, "r") as file:
            self.assertTrue(np.isnan(file["chunks/000002/temporal_mmd/mmd2"][()]))
            self.assertTrue(np.isfinite(file["chunks/000002/candidate_diversity/score"][()]))

    def test_guided_replay_captures_second_sampling_call(self):
        self.run_replay(guided=True)

    def test_explicit_noise_bypasses_rng(self):
        noise = torch.ones(15, 100, 32)
        self.policy.model.sample_noise = Mock(side_effect=AssertionError("Unexpected random draw"))
        self.policy.predict_action_chunk(self.batch, num_samples=15, noise=noise)
        np.testing.assert_array_equal(self.policy._replay_sampling_calls[0]["inputs"]["noise"], noise.numpy())

    def test_intervention_changes_keep_model_and_initialize_vlm_once(self):
        model = self.policy.model
        self.policy.reset = Mock()
        modes = [
            ("PIVOT", False, False), ("primitive", False, False), ("ensemble", False, False),
            ("none", True, False), ("none", False, True), ("none", False, False),
        ]
        for strategy, manual, spreads in modes:
            self.policy.reset_rollout()
            settings = {"interventions": strategy, "manual_guidance": manual, "vis_spreads": spreads,
                        "vlm_server_url": "http://127.0.0.1:35959"}
            self.policy.configure_replay_recording({"intervention_settings": settings})
            self.assertIs(self.policy.model, model)
            self.assertEqual(self.policy.interventions, False if strategy == "none" else strategy)
            self.assertEqual(self.policy.manual_guidance, manual)
            self.assertEqual(self.policy.vis_spreads, spreads)
            self.assertEqual(self.policy._rollout_config["intervention_settings"], settings)
        self.namespace["VLMClient"].assert_called_once_with(
            server_url="http://127.0.0.1:35959", model_name="Qwen/Qwen2.5-VL-72B-Instruct")

    def test_conflicting_intervention_settings_fail_before_vlm_initialization(self):
        for settings in (
            {"interventions": "invalid"},
            {"interventions": "PIVOT", "manual_guidance": True},
            {"interventions": "primitive", "vis_spreads": True},
            {"manual_guidance": True, "vis_spreads": True},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.policy.configure_interventions(**settings)
        self.namespace["VLMClient"].assert_not_called()
        self.assertFalse(self.policy.interventions)

    def test_reset_rollout_clears_recording_state(self):
        model = self.policy.model
        self.policy.count = 8
        self.policy.reset = Mock()
        self.policy.reset_rollout()
        self.policy.reset.assert_called_once()
        self.assertIs(self.policy.model, model)
        self.assertEqual(self.policy.count, 0)
        self.assertIsNone(self.policy._trajectory_recorder)
        self.assertEqual(self.policy._replay_sampling_calls, [])
        self.assertEqual(self.policy._raw_replay_observation, {})
        self.policy.configure_replay_recording({"robot": {"record": "next"}})
        self.assertEqual(self.policy._rollout_config["robot"]["record"], "next")

    def test_checkpoint_hash_is_reused_across_rollout_recordings(self):
        checkpoint = Path(self.directory.name) / "model.safetensors"
        checkpoint.write_bytes(b"loaded model weights")
        expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.policy._checkpoint_file = str(checkpoint)
        self.policy.config = make_dataclass("Config", [("n_action_steps", int)])(50)
        self.policy.reset = Mock()
        del self.policy._ensure_trajectory_recorder
        with patch("builtins.open", wraps=open) as read_file:
            for demo in ("first", "second"):
                self.policy.reset_rollout()
                self.policy.configure_replay_recording({"robot": {"record": demo}})
                with patch.dict(os.environ, {"LEROBOT_TRAJECTORY_DIR": self.directory.name,
                                            "LEROBOT_DEMO_NAME": demo}):
                    self.policy._ensure_trajectory_recorder()
                with h5py.File(self.policy._trajectory_recorder.path, "r") as file:
                    metadata = json.loads(file.attrs["metadata_json"])
                    self.assertEqual(metadata["checkpoint_sha256"], expected_hash)
                    self.assertEqual(metadata["rollout_config"]["robot"]["record"], demo)
            checkpoint_reads = [call for call in read_file.call_args_list if call.args[0] == str(checkpoint)]
            self.assertEqual(len(checkpoint_reads), 1)

    def test_full_rollout_configuration_and_launch_metadata_are_saved(self):
        self.policy._trajectory_recorder = None
        self.policy._checkpoint_file = None
        self.policy.config = make_dataclass("Config", [("n_action_steps", int)])(50)
        del self.policy._ensure_trajectory_recorder  # Use the real initializer for this test.
        config = {
            "dataset": {"single_task": "place both blocks in the bin", "fps": 30, "num_episodes": 1},
            "robot": {"record": "metadata", "port": "dummy"},
            "policy": {"n_action_steps": 50},
            "intervention_settings": {"interventions": "none", "vis_spreads": True},
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
            self.assertEqual(metadata["taco_settings"]["max_chunks"], 16)
            self.assertFalse(metadata["interventions"])
            self.assertFalse(metadata["manual_guidance"])
            self.assertTrue(metadata["vis_spreads"])
            for filename in ("collect_eval.bash", "pi_05_inference.bash", "lerobot_record.py", "franka.py"):
                self.assertIn(filename, file["artifacts/source"])
        with self.assertRaises(RuntimeError):
            self.policy.configure_replay_recording(config)


class TestFullHorizonReconstructionGuidance(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((POLICY_DIR / "modelling_pi05_taco.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI05Pytorch")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_actions")
        namespace = dict(torch=torch, Tensor=torch.Tensor, F=torch.nn.functional, copy=copy,
                         make_att_2d_masks=lambda pad, att: torch.ones(1, 1, 1, dtype=torch.bool))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "sampler_method", "exec"), namespace)
        self.sample = namespace["sample_actions"]
        self.noise = torch.ones(1, 100, 32)
        self.model = SimpleNamespace(
            config=SimpleNamespace(chunk_size=100, num_inference_steps=1),
            embed_prefix=Mock(return_value=(torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool), torch.ones(1))),
            _prepare_attention_masks_4d=lambda mask: mask,
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
                forward=Mock(return_value=(None, None)),
            ),
            denoise_step=Mock(return_value=torch.zeros_like(self.noise)),
        )

    def test_guidance_affects_every_predicted_step_independent_of_execution(self):
        for steps in (25, 50, 100):
            self.model.config.n_action_steps = steps
            result = self.sample(self.model, [], [], torch.zeros(1, 1), None,
                                 noise=self.noise.clone(), guidance_actions=torch.zeros(1, 100, 8))
            # One integration step cancels the residual on all 100 real action steps.
            torch.testing.assert_close(result[..., :8], torch.zeros(1, 100, 8))
            torch.testing.assert_close(result[..., 8:], self.noise[..., 8:])

    def test_short_long_and_malformed_guidance_are_rejected_before_inference(self):
        for shape in ((1, 0, 8), (1, 25, 8), (1, 50, 8), (1, 99, 8), (1, 101, 8), (100, 8)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "full prediction horizon"):
                self.sample(self.model, [], [], torch.zeros(1, 1), None,
                            noise=self.noise, guidance_actions=torch.zeros(shape))
        self.model.embed_prefix.assert_not_called()


if __name__ == "__main__":
    unittest.main()
