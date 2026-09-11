"""Runnable with unittest without loading the model or robot stack."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch


# pi05/__init__.py eagerly imports the model and its optional hardware dependencies.
MODULE_PATH = Path(__file__).resolve().parents[3] / "src/lerobot/policies/pi05/trajectory_recorder.py"
spec = importlib.util.spec_from_file_location("trajectory_recorder", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TrajectoryRecorder = module.TrajectoryRecorder


class TestMmdRbf(unittest.TestCase):
    def test_fixed_gamma_matches_reference_with_unequal_sample_counts(self):
        from sklearn.metrics.pairwise import rbf_kernel

        rng = np.random.default_rng(42)
        x, y = rng.normal(size=(15, 400)), rng.normal(size=(11, 400))
        for setting in (0.002, "median", "max_eig"):
            score, gamma = module.compute_mmd_rbf(x, y, setting)
            diversity, _ = module.compute_diversity_rbf(x, gamma)
            z = np.vstack((x, y))
            if setting == "median":
                distances = np.sum((z[:, None] - z[None]) ** 2, axis=-1)
                self.assertAlmostEqual(gamma, 1 / (2 * np.median(distances[distances > 0])))
            elif setting == "max_eig":
                self.assertAlmostEqual(gamma, 1 / np.linalg.eigvalsh(np.cov(z.T))[-1])
            expected = rbf_kernel(x, x, gamma).mean() + rbf_kernel(y, y, gamma).mean()
            expected -= 2 * rbf_kernel(x, y, gamma).mean()
            self.assertAlmostEqual(score, expected, places=12)
            self.assertAlmostEqual(diversity, 1 - rbf_kernel(x, x, gamma).mean(), places=12)

    def test_identical_and_constant_samples(self):
        x = np.arange(24).reshape(3, 8)
        for setting in (0.2, "median", "max_eig"):
            self.assertAlmostEqual(module.compute_mmd_rbf(x, x[::-1], setting)[0], 0)
            score, gamma = module.compute_mmd_rbf(np.ones((3, 8)), np.ones((3, 8)), setting)
            diversity, _ = module.compute_diversity_rbf(np.ones((3, 8)), setting)
            self.assertEqual(score, 0)
            self.assertEqual(diversity, 0)
            self.assertTrue(np.isfinite(gamma))

    def test_temporal_order_is_preserved(self):
        x = np.array([[0., 1., 2.]])
        score, _ = module.compute_mmd_rbf(x, x[:, ::-1], 0.5)
        self.assertAlmostEqual(score, 2 * (1 - np.exp(-4)))

    def test_diversity_increases_with_spread_at_fixed_gamma(self):
        spread = np.arange(5)[:, None] * 100.0
        identical = np.zeros((5, 1))
        score, _ = module.compute_mmd_rbf(spread, identical, 1.0)
        diversity, _ = module.compute_diversity_rbf(spread, 1.0)
        reversed_score, _ = module.compute_mmd_rbf(identical, spread, 1.0)
        reversed_diversity, _ = module.compute_diversity_rbf(identical, 1.0)
        self.assertAlmostEqual(score, reversed_score)
        self.assertAlmostEqual(diversity, 1 - 1 / 5)
        self.assertEqual(reversed_diversity, 0)

    def test_diversity_without_previous_chunk_including_single_candidate(self):
        from sklearn.metrics.pairwise import rbf_kernel

        for x in (np.arange(12).reshape(3, 4), np.ones((1, 4))):
            for setting in (0.2, "median", "max_eig"):
                diversity, gamma = module.compute_diversity_rbf(x, setting)
                self.assertTrue(np.isfinite(gamma))
                self.assertAlmostEqual(diversity, 1 - rbf_kernel(x, x, gamma).mean())

    def test_invalid_gamma_and_inputs(self):
        x = np.ones((2, 3))
        for gamma in (0, -1, np.inf, np.nan, "invalid"):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                module.compute_mmd_rbf(x, x, gamma)
        for y in (np.ones(3), np.ones((2, 4)), np.ones((0, 3)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                module.compute_mmd_rbf(x, y)


class TestTrajectoryRecorder(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        original = np.arange(15 * 50 * 8, dtype=np.float32).reshape(15, 50, 8)
        perturbed = original.copy()
        perturbed[:, 1:, :3] += 0.25
        self.record = dict(
            chunk_index=0,
            original=original,
            perturbed=perturbed,
            selected_indices=np.array([7, 0, 14, 3, 10]),
            observations={
                f"observation.images.camera_{camera}": np.full((1, 3, 8, 8), value, dtype=np.float32)
                for camera, value in (("front", 0.1), ("wrist", 0.5), ("side", 0.9))
            },
            task=["place both blocks in the bin"],
            perturb_std=0.01,
            raw_observation={"observation.state": np.arange(8, dtype=np.float32)[None]},
            perturbation_noise=np.ones((15, 49, 3), dtype=np.float32) * 25,
        )
        self.record["observations"].update({
            "observation.state": np.arange(8, dtype=np.float32)[None] / 10,
            "observation.language.tokens": np.array([[1, 42, 123456]], dtype=np.int64),
            "observation.language.attention_mask": np.array([[True, True, False]]),
        })

    def make_recorder(self):
        return TrajectoryRecorder({"checkpoint": "/checkpoints/003000/pretrained_model"}, self.directory.name, demo_name="test")

    def assert_record(self, chunk):
        self.assertTrue(chunk.attrs["complete"])
        np.testing.assert_array_equal(chunk["original_actions"], self.record["original"])
        np.testing.assert_array_equal(chunk["perturbed_actions"], self.record["perturbed"])
        np.testing.assert_array_equal(chunk["selected_indices"], self.record["selected_indices"])
        np.testing.assert_array_equal(
            chunk["selected_actions"], self.record["perturbed"][self.record["selected_indices"]]
        )
        self.assertEqual(len(chunk["observations"]), len(self.record["observations"]))
        for key, expected in self.record["observations"].items():
            np.testing.assert_array_equal(chunk["observations"][key], expected)
        self.assertEqual(json.loads(chunk.attrs["task_json"]), self.record["task"])
        np.testing.assert_array_equal(chunk["raw_observation/observation.state"], self.record["raw_observation"]["observation.state"])
        self.assertEqual(chunk["observations/observation.language.tokens"].dtype, np.dtype("int64"))
        self.assertEqual(chunk["observations/observation.language.attention_mask"].dtype, np.dtype("bool"))

    def test_chunks_are_readable_without_shutdown_and_existing_demo_is_overwritten(self):
        recorder = self.make_recorder()
        recorder.append(**self.record)
        with h5py.File(recorder.path, "r") as file:
            self.assert_record(file["chunks/000000"])
            self.assertIn("checkpoint", json.loads(file.attrs["metadata_json"]))
        recorder.append(**{**self.record, "chunk_index": 1})
        with h5py.File(recorder.path, "r") as file:
            self.assertEqual(len(file["chunks"]), 2)
            self.assert_record(file["chunks/000001"])
        self.assertEqual(recorder.path.name, "trajectories_test.h5")
        recorder = self.make_recorder()
        with h5py.File(recorder.path, "r") as file:
            self.assertEqual(len(file["chunks"]), 0)
        recorder.append(**self.record)
        with h5py.File(recorder.path, "r") as file:
            self.assertEqual(len(file["chunks"]), 1)
            self.assert_record(file["chunks/000000"])

    def test_demo_name_from_launcher_environment(self):
        with patch.dict(os.environ, {"LEROBOT_DEMO_NAME": "asdf"}):
            recorder = TrajectoryRecorder({}, self.directory.name)
        self.assertEqual(recorder.path.name, "trajectories_asdf.h5")

    def test_sigint_during_write_finishes_chunk_and_restores_handler(self):
        recorder = self.make_recorder()
        handler = signal.getsignal(signal.SIGINT)
        create_dataset = h5py.Group.create_dataset
        interrupted = False

        def interrupt_during_write(group, name, *args, **kwargs):
            nonlocal interrupted
            dataset = create_dataset(group, name, *args, **kwargs)
            if not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGINT)
            return dataset

        with patch.object(h5py.Group, "create_dataset", interrupt_during_write):
            with self.assertRaises(KeyboardInterrupt):
                recorder.append(**self.record)
        self.assertEqual(signal.getsignal(signal.SIGINT), handler)
        with h5py.File(recorder.path, "r") as file:
            self.assert_record(file["chunks/000000"])

    def test_tensor_indices_are_copied_to_cpu_before_numpy_conversion(self):
        recorder = self.make_recorder()
        indices = torch.from_numpy(self.record["selected_indices"])
        original_cpu = torch.Tensor.cpu
        copied_to_cpu = []

        def track_cpu(tensor, *args, **kwargs):
            copied_to_cpu.append(tensor)
            return original_cpu(tensor, *args, **kwargs)

        # Exercise the required transfer even on machines without a CUDA device.
        with patch.object(torch.Tensor, "cpu", track_cpu):
            recorder.append(**{**self.record, "selected_indices": indices})
        self.assertEqual(len(copied_to_cpu), 1)
        with h5py.File(recorder.path, "r") as file:
            self.assert_record(file["chunks/000000"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_indices_are_saved(self):
        recorder = self.make_recorder()
        indices = torch.tensor(self.record["selected_indices"], device="cuda")
        recorder.append(**{**self.record, "selected_indices": indices})
        with h5py.File(recorder.path, "r") as file:
            self.assert_record(file["chunks/000000"])

    def test_write_failure_preserves_previous_chunk(self):
        recorder = self.make_recorder()
        recorder.append(**self.record)
        with patch.object(h5py.Group, "create_dataset", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                recorder.append(**{**self.record, "chunk_index": 1})
        with h5py.File(recorder.path, "r") as file:
            self.assert_record(file["chunks/000000"])
            self.assertFalse(file["chunks/000001"].attrs["complete"])

    def test_online_mmd_uses_full_overlap_excludes_gripper_and_padding_and_resets(self):
        recorder = TrajectoryRecorder({"policy_config": {"mmd_gamma": 0.002}},
                                      self.directory.name, demo_name="mmd")
        rng = np.random.default_rng(8)
        previous = rng.normal(size=(15, 100, 32))
        current = rng.normal(size=(15, 100, 32))
        current[:, :50, :7] = previous[:, 50:, :7]
        # Matching overlaps must score zero despite different gripper, prefixes, tails, and padding.
        for index, full in enumerate((previous, current)):
            recorder.append(**{**self.record, "chunk_index": index,
                               "sampling_calls": [{"full_output": full}]})
        with h5py.File(recorder.path, "r") as file:
            first, second = (file[f"chunks/{index:06d}/temporal_mmd"] for index in (0, 1))
            self.assertTrue(np.isnan(first["mmd2"][()]))
            self.assertEqual(first["previous_record_index"][()], -1)
            self.assertAlmostEqual(second["mmd2"][()], 0)
            self.assertEqual(second["gamma"][()], 0.002)
            self.assertEqual(second["overlap_steps"][()], 50)
            self.assertEqual(second["num_samples"][()], 15)
            self.assertEqual(second["action_dim"][()], 7)
            self.assertEqual(second["previous_record_index"][()], 0)
        recorder.reset()
        recorder.append(**{**self.record, "sampling_calls": [{"full_output": current}]})
        with h5py.File(recorder.path, "r") as file:
            self.assertTrue(np.isnan(file["chunks/000002/temporal_mmd/mmd2"][()]))

    def test_no_overlap_is_undefined(self):
        recorder = self.make_recorder()
        for _ in range(2):
            recorder.append(**{**self.record, "sampling_calls": [{"full_output": self.record["original"]}]})
        with h5py.File(recorder.path, "r") as file:
            for chunk in file["chunks"].values():
                self.assertEqual(chunk["temporal_mmd/overlap_steps"][()], 0)
                self.assertTrue(np.isnan(chunk["temporal_mmd/mmd2"][()]))
                self.assertEqual(chunk["candidate_diversity/horizon"][()], 50)
                self.assertTrue(np.isfinite(chunk["candidate_diversity/score"][()]))

    def test_diversity_gamma_is_fixed_and_independent_of_previous_chunk(self):
        from sklearn.metrics.pairwise import rbf_kernel

        fixed_gamma = 0.002
        recorder = TrajectoryRecorder({"policy_config": {"mmd_gamma": "median", "diversity_gamma": fixed_gamma}},
                                      self.directory.name, demo_name="fixed_diversity")
        rng = np.random.default_rng(20)
        previous = rng.normal(size=(15, 100, 32))
        current = rng.normal(size=(15, 100, 32))
        for earlier in (previous, previous + 100):
            recorder.reset()
            for full in (earlier, current):
                recorder.append(**{**self.record, "sampling_calls": [{"full_output": full}]})
        x = current[:, :50, :7].reshape(15, -1)
        expected = 1 - rbf_kernel(x, x, gamma=fixed_gamma).mean()
        with h5py.File(recorder.path, "r") as file:
            for chunk in file["chunks"].values():
                self.assertEqual(chunk["candidate_diversity/gamma"][()], fixed_gamma)
            for index in (1, 3):
                self.assertAlmostEqual(file[f"chunks/{index:06d}/candidate_diversity/score"][()], expected)
            self.assertNotEqual(file["chunks/000001/temporal_mmd/gamma"][()],
                                file["chunks/000003/temporal_mmd/gamma"][()])

    def test_metrics_ignore_gripper_and_padding_but_include_orientation(self):
        recorder = self.make_recorder()
        full = np.zeros((15, 100, 32), dtype=np.float32)
        for index in range(3):
            if index == 1:
                full[:, :, 7:] = np.linspace(-1, 1, 15)[:, None, None]
            elif index == 2:
                full[:, :, 3] = np.linspace(-1, 1, 15)[:, None]
            recorder.append(**{**self.record, "chunk_index": index,
                               "original": full[:, :50, :8], "perturbed": full[:, :50, :8],
                               "sampling_calls": [{"full_output": full}]})
        with h5py.File(recorder.path, "r") as file:
            self.assertEqual(file["chunks/000000/candidate_diversity/score"][()], 0)
            self.assertEqual(file["chunks/000001/candidate_diversity/score"][()], 0)
            self.assertGreater(file["chunks/000002/candidate_diversity/score"][()], 0)
            self.assertEqual(file["chunks/000001/temporal_mmd/mmd2"][()], 0)
            self.assertEqual(file["chunks/000001/temporal_mmd/gamma"][()], 1)
            self.assertGreater(file["chunks/000002/temporal_mmd/mmd2"][()], 0)
            self.assertEqual(file["chunks/000001/temporal_mmd/action_dim"][()], 7)
            self.assertEqual(file["chunks/000001/candidate_diversity/action_dim"][()], 7)


if __name__ == "__main__":
    unittest.main()
