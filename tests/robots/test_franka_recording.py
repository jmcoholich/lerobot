"""Check recording lifecycle with stub hardware and no OpenTeach imports."""

import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open


ROBOT_DIR = Path(__file__).resolve().parents[2] / "src/lerobot/robots/franka"


class StubRobotConfig:
    @classmethod
    def register_subclass(cls, name):
        return lambda subclass: subclass


class StubRobot:
    def __init__(self, config):
        self.config = config


def load_class(filename, name, namespace):
    # Execute the real class body while avoiding imports of the hardware stack.
    tree = ast.parse((ROBOT_DIR / filename).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class TestFrankaRecording(unittest.TestCase):
    def setUp(self):
        self.operator_factory = Mock()
        namespace = dict(
            RobotConfig=StubRobotConfig, dataclass=dataclass, field=field, CameraConfig=object,
            Robot=StubRobot, DEBUG=False, DELTA_JOINT_ACTIONS=False, JOINT_ACTIONS=False,
            ZMQCameraSubscriber=Mock(), FrankaArmOperator=self.operator_factory,
            CONFIG_ROOT="/unused", os=os, EasyDict=dict,
            yaml=SimpleNamespace(safe_load=lambda file: {"host_address": "unused"}),
            open=mock_open(), print=Mock(),
        )
        self.config_class = load_class("franka_config.py", "FrankaConfig", namespace)
        self.robot_class = load_class("franka.py", "FrankaRobot", namespace)

    def test_record_name_reaches_operator_and_disconnect_saves_once(self):
        robot = self.robot_class(self.config_class(port="dummy", record="asdf"))
        self.assertEqual(self.operator_factory.call_args.kwargs["record"], "asdf")
        robot.disconnect()
        robot.disconnect()
        robot.operator.save_obs_cmd_history.assert_called_once_with()

    def test_default_record_name(self):
        robot = self.robot_class(self.config_class(port="dummy"))
        self.assertEqual(self.operator_factory.call_args.kwargs["record"], "last_recording")

    def test_failed_save_can_be_retried(self):
        robot = self.robot_class(self.config_class(port="dummy", record="retry"))
        robot.operator.save_obs_cmd_history.side_effect = [OSError("disk full"), None]
        with self.assertRaises(OSError):
            robot.disconnect()
        self.assertFalse(robot._saved_obs_cmd_history)
        robot.disconnect()
        self.assertTrue(robot._saved_obs_cmd_history)
        self.assertEqual(robot.operator.save_obs_cmd_history.call_count, 2)

if __name__ == "__main__":
    unittest.main()
