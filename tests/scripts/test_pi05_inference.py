"""Verify the inference-only client without connecting to hardware."""

import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from lerobot.scripts import pi05_inference as inference


class TestInferenceClient(unittest.TestCase):
    def test_import_and_cli_do_not_load_training_or_recording_dependencies(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from lerobot.scripts.pi05_inference import parse_args
parse_args(['--policy.path=/checkpoint', '--task=test', '--policy.dtype=bfloat16'])
for name in ('torch', 'torchvision', 'transformers', 'datasets', 'lerobot.processor',
             'lerobot.datasets.lerobot_dataset', 'lerobot.scripts.lerobot_record'):
    assert name not in sys.modules, name
""",
            ],
            check=True,
            timeout=30,
        )

    def test_policy_overrides_are_forwarded_without_loading_a_checkpoint(self):
        args = inference.parse_args(
            [
                "--policy.path=/not/a/local/checkpoint",
                "--task=place blocks",
                "--robot.record=trial",
                "--policy.dtype=bfloat16",
                "--policy.n_action_steps=25",
                "--policy.num_inference_steps=4",
            ]
        )
        self.assertEqual(args.policy_path, "/not/a/local/checkpoint")
        self.assertEqual(
            args.policy_overrides, ["--dtype=bfloat16", "--n_action_steps=25", "--num_inference_steps=4"]
        )
        self.assertEqual(args.record, "trial")

    def test_unknown_arguments_are_rejected(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            inference.parse_args(["--policy.path=/checkpoint", "--task=test", "--dataset.video=false"])

    def test_intervention_arguments_and_conflicts(self):
        base = ["--policy.path=/checkpoint", "--task=test"]
        for strategy in ("none", "PIVOT", "primitive", "ensemble"):
            args = inference.parse_args(base + [f"--interventions={strategy}"])
            self.assertEqual(args.interventions, strategy)
            self.assertEqual(args.policy_overrides, [])
        args = inference.parse_args(base + ["--manual_guidance=true"])
        self.assertTrue(args.manual_guidance)
        self.assertFalse(args.vis_spreads)
        args = inference.parse_args(base + ["--vis_spreads=true"])
        self.assertTrue(args.vis_spreads)
        self.assertFalse(args.manual_guidance)
        for invalid in (
            ["--interventions=unknown"],
            ["--manual_guidance=invalid"],
            ["--interventions=PIVOT", "--manual_guidance=true"],
            ["--manual_guidance=true", "--vis_spreads=true"],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit), patch("sys.stderr"):
                inference.parse_args(base + invalid)

    def test_commands_and_history_cleanup_on_policy_exit_or_disconnect(self):
        args = inference.parse_args(
            [
                "--policy.path=/checkpoint",
                "--task=test",
                "--robot.record=trial",
                "--interventions=ensemble",
            ]
        )
        for failure in (SystemExit(0), ConnectionError("disconnected"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                robot = Mock(robot_type="franka", is_connected=True)
                robot.action_features = {"x": float, "gripper": float}
                robot.observation_features = {"joint": float}
                robot.get_observation.return_value = {"joint": 0.5}
                client = Mock()
                action = {"x": 0.1, "gripper": 0.2}
                client.predict_action.side_effect = [action, failure]
                with (
                    patch.object(inference, "FrankaRobot", return_value=robot),
                    patch.object(inference, "RemotePolicyClient", return_value=client),
                    patch.object(inference, "precise_sleep"),
                    self.assertRaises(type(failure)),
                ):
                    inference.run(args)
                robot.send_action.assert_called_once_with(action)
                robot.disconnect.assert_called_once()
                client.close.assert_called_once()
                client.predict_action.assert_called_with({"joint": 0.5}, "test", robot, raw_observation=True)
                config = client.request.call_args.kwargs
                self.assertEqual(config["rollout_config"]["robot"]["record"], "trial")
                self.assertEqual(config["rollout_config"]["dataset"]["single_task"], "test")
                self.assertEqual(
                    config["rollout_config"]["intervention_settings"],
                    {
                        "interventions": "ensemble",
                        "manual_guidance": False,
                        "vis_spreads": False,
                    },
                )


if __name__ == "__main__":
    unittest.main()
