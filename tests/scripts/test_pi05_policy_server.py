"""Exercise cache and socket lifecycle without weights or robot dependencies."""

import copy
import os
import tempfile
import threading
import unittest
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.scripts.pi05_policy_server import PolicySession, RemotePolicyClient, serve_connection


@dataclass
class Config:
    type: str = "pi05"
    pretrained_path: str = "/checkpoint"
    n_action_steps: int = 50
    dtype: str = "bfloat16"
    device: str = "cpu"


class TestPolicySession(unittest.TestCase):
    def setUp(self):
        self.factory = SimpleNamespace(
            make_policy=Mock(side_effect=lambda cfg, **kw: Mock(config=cfg)),
            make_pre_post_processors=Mock(side_effect=lambda **kw: (Mock(), Mock())),
        )
        self.modules = patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                "lerobot.policies.factory": self.factory,
                "lerobot.processor.rename_processor": SimpleNamespace(
                    rename_stats=lambda stats, rename: stats
                ),
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.session = PolicySession()
        self.setup = {
            "policy_cfg": Config(),
            "dataset_meta": SimpleNamespace(features={}, stats={}),
            "rename_map": {},
            "rollout_config": {"dataset": {"single_task": "first task"}},
            "environment": {"LEROBOT_DEMO_NAME": "first", "LEROBOT_TRAJECTORY_DIR": "/tmp/first"},
        }

    def configure(self, **overrides):
        self.session.configure(**(copy.deepcopy(self.setup) | overrides))

    def test_new_rollout_reuses_weights_and_processors_but_resets_state(self):
        self.configure()
        policy = self.session.policy
        self.configure(
            rollout_config={"dataset": {"single_task": "new task"}},
            environment={"LEROBOT_DEMO_NAME": "second"},
        )
        self.factory.make_policy.assert_called_once()
        self.factory.make_pre_post_processors.assert_called_once()
        self.assertIs(self.session.policy, policy)
        self.assertEqual(policy.reset_rollout.call_count, 2)
        self.assertEqual(self.session.preprocessor.reset.call_count, 2)
        policy.configure_replay_recording.assert_called_with({"dataset": {"single_task": "new task"}})
        self.assertEqual(os.environ["LEROBOT_DEMO_NAME"], "second")
        self.assertNotIn("LEROBOT_TRAJECTORY_DIR", os.environ)

    def test_intervention_settings_are_forwarded_without_reloading(self):
        for strategy in ("none", "PIVOT", "primitive", "ensemble", "none"):
            rollout = {"intervention_settings": {"interventions": strategy}}
            self.configure(rollout_config=rollout)
            self.session.policy.configure_replay_recording.assert_called_with(rollout)
        self.factory.make_policy.assert_called_once()
        self.factory.make_pre_post_processors.assert_called_once()

    def test_checkpoint_and_inference_changes_reload(self):
        self.configure()
        for field, value in (
            ("pretrained_path", "/other"),
            ("n_action_steps", 20),
            ("dtype", "float32"),
            ("device", "cuda"),
        ):
            setattr(self.setup["policy_cfg"], field, value)
            self.configure()
        self.assertEqual(self.factory.make_policy.call_count, 5)

    def test_failed_processor_load_is_retried(self):
        self.factory.make_pre_post_processors.side_effect = RuntimeError("broken processor")
        with self.assertRaisesRegex(RuntimeError, "broken processor"):
            self.configure()
        self.assertIsNone(self.session.cache_key)
        self.factory.make_pre_post_processors.side_effect = lambda **kw: (Mock(), Mock())
        self.configure()
        self.assertEqual(self.factory.make_policy.call_count, 2)


class TestConnections(unittest.TestCase):
    def test_raw_protocol_returns_named_actions_and_stops_only_the_rollout(self):
        session = Mock()
        session.predict_robot.side_effect = [{"x": 0.25}, SystemExit(0)]
        robot = SimpleNamespace(robot_type="franka", debug=True)
        with tempfile.TemporaryDirectory() as directory:
            address = os.path.join(directory, "policy.sock")
            with Listener(address, family="AF_UNIX") as listener:

                def serve():
                    with listener.accept() as connection:
                        serve_connection(connection, session)

                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                client = RemotePolicyClient(address)
                try:
                    client.request("configure_robot", policy_path="/checkpoint")
                    action = client.predict_action({"joint": 1.0}, "task", robot, raw_observation=True)
                    self.assertEqual(action, {"x": 0.25})
                    with self.assertRaises(SystemExit):
                        client.predict_action({"joint": 1.0}, "task", robot, raw_observation=True)
                finally:
                    client.close()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                session.configure_robot.assert_called_once_with(policy_path="/checkpoint")

    def test_client_sends_only_robot_pose_for_guidance(self):
        client = RemotePolicyClient.__new__(RemotePolicyClient)
        client.request = Mock(return_value=[1, 2])
        interface = SimpleNamespace(last_eef_pose=[[1]], last_eef_quat_and_pos=([1], [2]))
        robot = SimpleNamespace(
            robot_type="franka", debug=False, operator=SimpleNamespace(robot_interface=interface)
        )
        with patch.dict("sys.modules", {"torch": SimpleNamespace(from_numpy=lambda value: value)}):
            self.assertEqual(client.predict_action({"observation.state": [0]}, "task", robot), [1, 2])
        request = client.request.call_args.kwargs
        self.assertEqual(request["task"], "task")
        self.assertEqual(request["observation"], {"observation.state": [0]})
        self.assertIsNot(request["robot"], robot)
        self.assertEqual(request["robot"].operator.robot_interface.last_eef_pose, [[1]])

    def test_disconnect_errors_and_policy_exit_allow_reconnection(self):
        session = Mock()
        session.reset.return_value = None
        session.predict.side_effect = [SystemExit(0), ValueError("inference failed"), "action"]
        with tempfile.TemporaryDirectory() as directory:
            address = os.path.join(directory, "policy.sock")
            with Listener(address, family="AF_UNIX") as listener:

                def serve():
                    for _ in range(4):
                        with listener.accept() as connection:
                            serve_connection(connection, session)

                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                # A recorder killed between calls simply disconnects.
                with Client(address, family="AF_UNIX"):
                    pass
                for error, expected in (
                    (SystemExit, 0),
                    (RuntimeError, "inference failed"),
                    (None, "action"),
                ):
                    client = RemotePolicyClient(address)
                    try:
                        client.request("configure")
                        client.reset()
                        if error:
                            with self.assertRaises(error) as raised:
                                client.request("predict")
                            if error is SystemExit:
                                self.assertEqual(raised.exception.code, expected)
                            else:
                                self.assertIn(expected, str(raised.exception))
                        else:
                            self.assertEqual(client.request("predict"), expected)
                    finally:
                        client.close()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
