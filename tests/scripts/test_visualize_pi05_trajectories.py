"""Grid layout and CPU normalization-artifact checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from safetensors.numpy import save


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/visualize_pi05_trajectories.py"
spec = importlib.util.spec_from_file_location("visualize_pi05_trajectories", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestTrajectoryVisualization(unittest.TestCase):
    def test_intervention_choices_for_each_strategy_and_missing_records(self):
        with h5py.File("choices.h5", "w", driver="core", backing_store=False) as file:
            chunk = file.create_group("chunk")
            self.assertEqual(module.intervention_detail_lines(chunk), [])
            group = chunk.create_group("intervention")
            group.create_dataset("occurred", data=True)
            group.attrs.update(pivot_color="Blue", primitive="rotate_ccw")
            for setting, expected in (
                ("PIVOT", ["PIVOT color: Blue"]),
                ("primitive", ["Primitive: rotate_ccw"]),
                ("ensemble", ["PIVOT color: Blue", "Primitive: rotate_ccw"]),
            ):
                group.attrs["setting"] = setting
                self.assertEqual(module.intervention_detail_lines(chunk), expected)
            del group.attrs["primitive"]
            self.assertEqual(module.intervention_detail_lines(chunk), ["PIVOT color: Blue", "Primitive: not recorded"])
            group["occurred"][()] = False
            self.assertEqual(module.intervention_detail_lines(chunk), [])

    def test_diversity_is_rendered_with_three_decimals_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.h5"
            with h5py.File(path, "w") as file:
                file.attrs["metadata_json"] = json.dumps({"interventions": "ensemble"})
                for index, diversity in enumerate((0.6784, np.nan, None)):
                    chunk = file.create_group(f"chunks/{index:06d}")
                    chunk.attrs["complete"] = True
                    chunk.create_dataset(module.FRONT_IMAGE, data=np.zeros((1, 3, 360, 640), dtype=np.float32))
                    chunk.create_dataset("selected_actions", data=np.zeros((5, 50, 8)))
                    chunk.create_dataset("selected_indices", data=np.arange(5))
                    chunk.create_dataset("sampling/000000/full_output", data=np.zeros((15, 100, 32)))
                    if diversity is not None:
                        chunk.create_dataset("candidate_diversity/score", data=diversity)
                    if index < 2:
                        chunk.create_dataset("intervention/occurred", data=index == 0)
                        chunk.create_dataset("intervention/triggered", data=True)
                        chunk.create_dataset("intervention/score", data=1.23456 if index == 0 else np.nan)
                        chunk.create_dataset("intervention/threshold", data=1.2082)
                        chunk["intervention"].attrs["setting"] = "ensemble"
                        chunk["intervention"].attrs.update(pivot_color="Blue", primitive="rotate_ccw")
                        chunk.create_dataset("timing/execution_s", data=2.0 if index == 0 else np.nan)
                        chunk.create_dataset("timing/base_inference_s", data=0.5)
                        chunk.create_dataset("timing/intervention_s", data=4.0)
                        chunk.create_dataset("timing/guided_inference_s", data=3.0)
            tile = np.full((360, 360, 3), 60, dtype=np.uint8)
            with patch.object(module, "load_plot_postprocessor", return_value=None), \
                 patch.object(module, "load_renderer", return_value=lambda *args, **kwargs: (tile, None)), \
                 patch.object(module.cv2, "putText", wraps=module.cv2.putText) as draw:
                output = module.create_visualization(path)
            labels = [call.args[1] for call in draw.call_args_list if call.args[1].startswith("Candidate diversity:")]
            self.assertEqual(labels, ["Candidate diversity: 0.678", "Candidate diversity: N/A",
                                      "Candidate diversity: not recorded"])
            interventions = [call.args[1] for call in draw.call_args_list if call.args[1].startswith("Intervention:")]
            self.assertEqual(interventions, ["Intervention: EVE", "Intervention: EVE"])
            self.assertEqual(sum(call.args[1] == "PIVOT color: Blue" for call in draw.call_args_list), 1)
            self.assertEqual(sum(call.args[1] == "Primitive: rotate_ccw" for call in draw.call_args_list), 1)
            scores = [call.args[1] for call in draw.call_args_list if call.args[1].startswith("Intervention score:")]
            self.assertEqual(scores, ["Intervention score: 1.235", "Intervention score: N/A", "Intervention score: not recorded"])
            self.assertIn("Intervention threshold: 1.2082", [call.args[1] for call in draw.call_args_list])
            times = [call.args[1] for call in draw.call_args_list if call.args[1].startswith("Execution + inference:")]
            self.assertEqual(times, ["Execution + inference: 2.500 s", "Execution + inference: N/A",
                                     "Execution + inference: not recorded"])
            intervention_times = [call.args[1] for call in draw.call_args_list if call.args[1].startswith("Intervention time:")]
            self.assertEqual(intervention_times, ["Intervention time: 4.000 s"])
            rendered = module.cv2.imread(str(output))
            # The badge is visible only when guidance actually occurred, not merely triggered.
            badge_y = 182 + module.GAP + 114
            np.testing.assert_array_equal(rendered[badge_y, module.GAP + 326], [0, 190, 255])
            np.testing.assert_array_equal(rendered[badge_y, module.GAP + 360 + module.GAP + 326], [24, 24, 24])
            np.testing.assert_array_equal(rendered[182 + module.GAP + module.LABEL_HEIGHT:
                                                  182 + module.GAP + module.LABEL_HEIGHT + 360,
                                                  module.GAP:module.GAP + 360], tile)

    def test_partial_execution_timing_is_labeled_and_missing_inference_is_not_zero(self):
        with h5py.File("timing.h5", "w", driver="core", backing_store=False) as file:
            chunk = file.create_group("chunk")
            chunk.create_dataset("timing/execution_s", data=1.25)
            chunk.create_dataset("timing/execution_complete", data=False)
            self.assertEqual(module.timing_label(chunk, "execution_s", "base_inference_s"), "not recorded")
            chunk.create_dataset("timing/base_inference_s", data=.5)
            self.assertEqual(module.timing_label(chunk, "execution_s", "base_inference_s"), "1.750 s (partial)")

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
        tiles = [(str(index), np.full((12, 24, 3), index + 1, dtype=np.uint8), np.nan, np.nan, "") for index in range(7)]
        grid = module.combine_tiles(tiles)
        self.assertEqual(grid.shape, (2 * (12 + module.LABEL_HEIGHT) + 3 * module.GAP, 6 * 24 + 7 * module.GAP, 3))
        for index, (_, tile, _, _, _) in enumerate(tiles):
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
