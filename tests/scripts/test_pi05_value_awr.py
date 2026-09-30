"""CPU-only end-to-end checks for OOF routing and dataset annotation."""

import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    "value_awr", Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_pi05_value_awr.py"
)
awr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awr)


class TestOOFAdvantages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.root = self.repo / "dataset"
        (self.root / "data/chunk-000").mkdir(parents=True)
        (self.root / "meta").mkdir()
        self.key = "annotation_return_gamma_0.95"
        # Interleave episodes and reverse global ids to exercise keyed rather than positional joins.
        self.table = pa.table(
            {
                "index": list(reversed(range(10))),
                "episode_index": list(range(5)) * 2,
                "frame_index": [0] * 5 + [1] * 5,
                self.key: np.arange(10, dtype=np.float32),
                "unrelated": ["preserve me"] * 10,
                "fname": [f"demo_walle_plug3_bc_and_dagger_direct_{ep}.h5" for ep in list(range(5)) * 2],
            }
        )
        hf = {
            "info": {
                "features": {
                    field.name: {"dtype": str(field.type), "_type": "Value"} for field in self.table.schema
                }
            }
        }
        self.table = self.table.replace_schema_metadata({b"huggingface": json.dumps(hf).encode()})
        self.files = [self.root / f"data/chunk-000/file-{i:03d}.parquet" for i in range(2)]
        for i, file in enumerate(self.files):
            pq.write_table(self.table.slice(i * 5, 5), file)
        awr.write_json(
            self.root / "meta/info.json", {"total_episodes": 5, "total_frames": 10, "fps": 20, "features": {}}
        )
        awr.write_json(self.root / "meta/stats.json", {})
        self.configs = []
        for fold in range(1, 6):
            path = self.repo / f"outputs/model_fold_{fold}/checkpoints/last/pretrained_model"
            path.mkdir(parents=True)
            cfg = {
                "use_value_model": True,
                "value_dim": 1,
                "value_key": self.key,
                "value_discount": 0.95,
                "value_reward_key": "annotation_reward",
                "value_bootstrap_steps": 0,
            }
            awr.write_json(path / "config.json", cfg)
            awr.write_json(
                path / "train_config.json",
                {
                    "policy": cfg,
                    "dataset": {"root": str(self.root)},
                    "train_episodes": [ep for ep in range(5) if ep != fold - 1],
                    "test_episodes": [fold - 1],
                },
            )
            (path / "model.safetensors").write_bytes(b"fixture")
            self.configs.append(path)
        self.args = argparse.Namespace(
            dataset_root=self.root,
            run_template="model_fold_{fold}",
            checkpoint="last",
            name=None,
            output_dir=self.repo / "reports",
            dry_run=False,
        )
        self.repo_patch = patch.object(awr, "REPO", self.repo)
        self.repo_patch.start()
        self.addCleanup(self.repo_patch.stop)
        self.plan = awr.prepare(self.args)
        self.output = self.repo / "predictions"
        self.output.mkdir()
        self.plan["output_dir"] = str(self.output)
        self.plan_path = self.output / "plan.json"
        awr.write_json(self.plan_path, self.plan)
        self.prediction_paths = []
        for fold in self.plan["folds"]:
            ep = fold["episodes"][0]
            path = self.output / fold["manifest"] / f"task_0/episode_{ep}/all_cams.json"
            path.parent.mkdir(parents=True)
            awr.write_json(
                path,
                {
                    fold["prediction_key"]: {
                        "model": fold["policy_path"],
                        "value_key": self.key,
                        "prediction_scale": "raw",
                        "unnormalized_from": "QUANTILES",
                        "frame_indices": [0, 1],
                        "values": [ep - 0.25, ep + 4.75],
                    },
                    self.key: {"source": str(self.root), "frame_indices": [0, 1], "values": [ep, ep + 5]},
                },
            )
            self.prediction_paths.append(path)

    def test_merge_preserves_rows_and_supports_huggingface_loading(self):
        from datasets import Dataset

        with contextlib.redirect_stdout(io.StringIO()):
            awr.merge(self.plan_path)
        table = pa.concat_tables([pq.read_table(path) for path in self.files])
        for key in self.table.column_names:
            self.assertEqual(table[key].to_pylist(), self.table[key].to_pylist())
        advantage, value = self.plan["columns"]["advantage"], self.plan["columns"]["value"]
        np.testing.assert_allclose(table[advantage].to_numpy(), 0.25)
        np.testing.assert_allclose(table[value].to_numpy(), np.arange(10) - 0.25)
        loaded = Dataset.from_parquet([str(path) for path in self.files])
        self.assertEqual(loaded.features[advantage].dtype, "float32")
        self.assertEqual(awr.read_json(self.root / "meta/info.json")["features"][advantage]["shape"], [1])
        self.assertEqual(awr.read_json(self.root / "meta/stats.json")[advantage]["count"], [10])
        self.assertTrue((self.output / "histograms.png").is_file())
        for episode in range(5):
            self.assertTrue((self.output / f"episode_plots/episode_{episode:06d}.png").is_file())
        self.assertEqual(awr.read_json(self.output / "stats.json")["value_error"]["mae"], 0.25)
        backup = self.root / "meta/awr_backups/model_last" / self.files[0].relative_to(self.root)
        self.assertEqual(pq.read_table(backup).column_names, self.table.column_names)
        with self.assertRaisesRegex(ValueError, "already exist"):
            awr.merge(self.plan_path)

    def test_missing_predictions_fail_before_dataset_changes(self):
        original = self.files[0].read_bytes()
        self.prediction_paths[-1].unlink()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            awr.merge(self.plan_path)
        self.assertEqual(self.files[0].read_bytes(), original)

    def test_wrong_returns_and_sampled_frames_are_rejected(self):
        _, rows = awr.dataset_rows(self.root, self.key)
        path = self.prediction_paths[0]
        data = awr.read_json(path)
        data[self.key]["values"][0] = 999
        awr.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "returns differ"):
            awr.collect(self.plan, rows)
        data[self.key]["values"][0] = 0
        data[self.plan["folds"][0]["prediction_key"]]["frame_indices"] = [0]
        awr.write_json(path, data)
        with self.assertRaisesRegex(ValueError, "frames"):
            awr.collect(self.plan, rows)

    def test_leakage_and_bootstrapped_targets_are_rejected(self):
        path = self.configs[0]
        train = awr.read_json(path / "train_config.json")
        train["train_episodes"].append(0)
        awr.write_json(path / "train_config.json", train)
        with self.assertRaisesRegex(ValueError, "disjoint partition"):
            awr.prepare(self.args)
        cfg = awr.read_json(path / "config.json")
        cfg["value_bootstrap_steps"] = 10
        awr.write_json(path / "config.json", cfg)
        with self.assertRaisesRegex(ValueError, "bootstrapped"):
            awr.prepare(self.args)

    def test_mixed_return_discounts_are_rejected(self):
        path = self.configs[1] / "config.json"
        cfg = awr.read_json(path)
        cfg["value_discount"] = 0.99
        cfg["value_key"] = "annotation_return_gamma_0.99"
        awr.write_json(path, cfg)
        with self.assertRaisesRegex(ValueError, "different return targets"):
            awr.prepare(self.args)

    def test_plan_exports_only_held_out_episodes_for_bash(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            awr.write_plan(self.args)
        output = Path(stdout.getvalue().strip())
        plan = awr.read_json(output / "plan.json")
        self.assertEqual(plan["output_dir"], str(output))
        rows = (output / "folds.tsv").read_text().splitlines()
        self.assertEqual(len(rows), 5)
        for ep, row in enumerate(rows):
            run, policy_path, manifest, episodes = row.split("\t")
            self.assertEqual(run, f"model_fold_{ep + 1}")
            self.assertEqual(policy_path, str(self.configs[ep]))
            self.assertEqual(manifest, f"fold_{ep + 1}")
            self.assertEqual(episodes, str(ep))

    def test_dry_run_does_not_write_a_plan(self):
        self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()):
            awr.write_plan(self.args)
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
