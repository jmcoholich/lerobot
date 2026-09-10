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

    def test_chunks_are_readable_without_shutdown_and_existing_demo_is_preserved(self):
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
        with self.assertRaises(FileExistsError):
            self.make_recorder()
        with h5py.File(recorder.path, "r") as file:
            self.assertEqual(len(file["chunks"]), 2)

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


if __name__ == "__main__":
    unittest.main()
