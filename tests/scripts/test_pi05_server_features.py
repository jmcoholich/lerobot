"""Compare inference-only feature conversion with the existing recording path."""

import tempfile
import unittest
from unittest.mock import patch


class TestServerFeatures(unittest.TestCase):
    def test_raw_observations_actions_overrides_and_cache_match_recording_path(self):
        import numpy as np
        import torch

        from lerobot.datasets.pipeline_features import (
            aggregate_pipeline_dataset_features,
            create_initial_features,
        )
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.processor import make_default_processors
        from lerobot.scripts.pi05_policy_server import PolicySession

        class Identity:
            def __call__(self, value):
                return value

            def reset(self):
                pass

        class DummyPolicy(torch.nn.Module):
            @classmethod
            def from_pretrained(cls, config, **kwargs):
                policy = cls()
                policy.config = config
                return policy

            def reset(self):
                pass

            def reset_rollout(self):
                pass

            def configure_replay_recording(self, config):
                self.rollout_config = config

            def select_action(self, batch, **kwargs):
                self.batch = batch
                return torch.tensor([[0.25, 0.75]])

        hw_features = {
            "action": {"x": float, "gripper": float},
            "observation": {"joint_b": float, "joint_a": float, "camera_front": (4, 6, 3)},
        }
        action_proc, _, obs_proc = make_default_processors()
        old_features = {
            **aggregate_pipeline_dataset_features(
                action_proc, create_initial_features(action=hw_features["action"])
            ),
            **aggregate_pipeline_dataset_features(
                obs_proc, create_initial_features(observation=hw_features["observation"])
            ),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("lerobot.policies.factory.get_policy_class", return_value=DummyPolicy) as loader,
            patch("lerobot.policies.factory.make_pre_post_processors", return_value=(Identity(), Identity())),
            patch.dict("os.environ"),
        ):
            PI05Config(device="cpu").save_pretrained(directory)
            session = PolicySession()
            for task in ("first", "second"):
                session.configure_robot(
                    directory,
                    ["--n_action_steps=3", "--num_inference_steps=4"],
                    hw_features,
                    {"dataset": {"single_task": task}},
                    {},
                )
            loader.assert_called_once()
            self.assertEqual(session.robot_features, old_features)
            self.assertEqual(session.policy.config.n_action_steps, 3)
            self.assertEqual(session.policy.config.num_inference_steps, 4)
            self.assertEqual(session.policy.rollout_config["dataset"]["single_task"], "second")
            action = session.predict_robot(
                {"joint_a": 2.0, "joint_b": 7.0, "camera_front": np.full((4, 6, 3), 255, dtype=np.uint8)},
                task="second",
                robot_type="franka",
                robot=None,
            )
            self.assertEqual(action, {"x": 0.25, "gripper": 0.75})
            torch.testing.assert_close(session.policy.batch["observation.state"], torch.tensor([[7.0, 2.0]]))
            torch.testing.assert_close(
                session.policy.batch["observation.images.camera_front"], torch.ones((1, 3, 4, 6))
            )


if __name__ == "__main__":
    unittest.main()
