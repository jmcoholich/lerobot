"""Grid layout and CPU normalization-artifact checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from safetensors.numpy import save


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/visualize_pi05_trajectories.py"
spec = importlib.util.spec_from_file_location("visualize_pi05_trajectories", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestTrajectoryVisualization(unittest.TestCase):
    def test_run_time_uses_saved_recording_date(self):
        with h5py.File("timestamp.h5", "w", driver="core", backing_store=False) as file:
            file.attrs["created_at_utc"] = "20260910T004638_105191Z"
            file.create_group("chunks/000000").attrs["timestamp_unix"] = 1789001198.8948574
            self.assertEqual(module.recorded_run_time(file), "September 09, 2026 at 08:46:38 PM EDT")

    def test_run_time_uses_standard_time_in_winter(self):
        with h5py.File("timestamp.h5", "w", driver="core", backing_store=False) as file:
            file.attrs["created_at_utc"] = "20260110T004638_105191Z"
            self.assertEqual(module.recorded_run_time(file), "January 09, 2026 at 07:46:38 PM EST")

    def test_run_time_falls_back_to_earliest_chunk_or_missing(self):
        with h5py.File("timestamp.h5", "w", driver="core", backing_store=False) as file:
            file.create_group("chunks")
            self.assertEqual(module.recorded_run_time(file), "not recorded")
            file.create_group("chunks/000000").attrs["timestamp_unix"] = 1789001198.8948574
            file.create_group("chunks/000001").attrs["timestamp_unix"] = 1789001298.0
            self.assertEqual(module.recorded_run_time(file), "September 09, 2026 at 08:46:38 PM EDT")

    def test_full_horizon_preserves_selected_prefix_and_sample_order(self):
        full = np.arange(15 * 100 * 32, dtype=np.float32).reshape(15, 100, 32)
        indices = np.array([9, 2, 12, 0, 7])
        selected = full[indices, :50, :8].copy()
        selected[:, 1:, :3] += 0.25
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "test.h5", "w") as file:
                chunk = file.create_group("chunks/000000")
                chunk.create_dataset("selected_actions", data=selected)
                chunk.create_dataset("selected_indices", data=indices)
                chunk.create_dataset("sampling/000000/full_output", data=full)
                actions = module.full_selected_trajectories(chunk)
                self.assertEqual(actions.shape, (5, 100, 8))
                np.testing.assert_array_equal(actions[:, :50], selected)
                np.testing.assert_array_equal(actions[:, 50:], full[indices, 50:, :8])
                np.testing.assert_array_equal(actions[:, 99], full[indices, 99, :8])
                np.testing.assert_array_equal(chunk["selected_actions"], selected)

    def test_seventh_image_starts_second_row_without_resizing(self):
        tiles = [(str(index), np.full((12, 24, 3), index + 1, dtype=np.uint8)) for index in range(7)]
        grid = module.combine_tiles(tiles)
        self.assertEqual(grid.shape, (2 * (12 + module.LABEL_HEIGHT) + 3 * module.GAP, 6 * 24 + 7 * module.GAP, 3))
        for index, (_, tile) in enumerate(tiles):
            row, column = divmod(index, 6)
            x = module.GAP + column * (24 + module.GAP)
            y = module.GAP + row * (12 + module.LABEL_HEIGHT + module.GAP) + module.LABEL_HEIGHT
            np.testing.assert_array_equal(grid[y:y + 12, x:x + 24], tile)
        # The unused slot in the last row stays background-colored.
        self.assertTrue(np.all(grid[-module.GAP - 12:-module.GAP, -module.GAP - 24:-module.GAP] == 24))

    def test_empty_grid_is_reported(self):
        with self.assertRaisesRegex(ValueError, "No complete trajectory"):
            module.combine_tiles([])

    def test_action_ranges_loaded_from_saved_processor(self):
        low, high = np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 2
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "test.h5", "w") as file:
                group = file.create_group("artifacts/postprocessor")
                config = json.dumps({"steps": [{"state_file": "stats.safetensors"}]}).encode()
                group.create_dataset("processor.json", data=np.frombuffer(config, dtype=np.uint8))
                group.create_dataset("stats.safetensors", data=np.frombuffer(save({"action.min": low, "action.max": high}), dtype=np.uint8))
                postprocessor = module.load_plot_postprocessor(file)
                np.testing.assert_array_equal(postprocessor.steps[0].stats["action"]["min"], low)
                np.testing.assert_array_equal(postprocessor.steps[0].stats["action"]["max"], high)


if __name__ == "__main__":
    unittest.main()
