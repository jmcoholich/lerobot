from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaConfig(RobotConfig):
    port: str
    record: str = "last_recording"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
