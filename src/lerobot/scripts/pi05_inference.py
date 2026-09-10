"""Franka policy-server client without LeRobot dataset recording or model imports."""

import argparse
import logging
import os
import time
from contextlib import closing
from dataclasses import asdict

from lerobot.robots.franka import FrankaConfig, FrankaRobot
from lerobot.scripts.pi05_policy_server import DEFAULT_SOCKET, ROLLOUT_ENV, RemotePolicyClient
from lerobot.utils.robot_utils import precise_sleep


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_server", default=DEFAULT_SOCKET)
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--robot.type", choices=["franka"], default="franka")
    parser.add_argument("--robot.id", dest="robot_id", default="franka")
    parser.add_argument("--robot.port", dest="robot_port", default="dummy")
    parser.add_argument("--robot.record", dest="record", default="last_recording")
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration", type=float, default=30000)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--interventions", choices=["none", "PIVOT", "primitive", "ensemble"], default="none")
    parser.add_argument("--manual_guidance", choices=["true", "false"], default="false")
    parser.add_argument("--vis_spreads", choices=["true", "false"], default="false")
    args, overrides = parser.parse_known_args(argv)
    args.manual_guidance = args.manual_guidance == "true"
    args.vis_spreads = args.vis_spreads == "true"
    if sum((args.interventions != "none", args.manual_guidance, args.vis_spreads)) > 1:
        parser.error("Choose only one of interventions, manual guidance, or spread visualization")
    if args.duration <= 0 or args.fps <= 0:
        parser.error("--duration and --fps must be positive")
    if any(not arg.startswith("--policy.") or "=" not in arg for arg in overrides):
        parser.error("Additional arguments must use --policy.NAME=VALUE")
    args.policy_overrides = [arg.replace("--policy.", "--", 1) for arg in overrides]
    return args


def run(args):
    with closing(RemotePolicyClient(args.policy_server)) as client:
        setup_t = time.perf_counter()
        robot_cfg = FrankaConfig(id=args.robot_id, port=args.robot_port, record=args.record)
        robot = FrankaRobot(robot_cfg)
        logging.info("Startup: robot setup %.2fs", time.perf_counter() - setup_t)
        try:
            setup_t = time.perf_counter()
            client.request(
                "configure_robot",
                policy_path=args.policy_path,
                policy_overrides=args.policy_overrides,
                robot_features={"action": robot.action_features, "observation": robot.observation_features},
                # Keep the replay metadata's task/FPS locations without creating a dataset.
                rollout_config={
                    "robot": asdict(robot_cfg),
                    "dataset": {
                        "single_task": args.task,
                        "fps": args.fps,
                        "episode_time_s": args.duration,
                        "num_episodes": 1,
                    },
                    "policy_server": args.policy_server,
                    "intervention_settings": {
                        "interventions": args.interventions,
                        "manual_guidance": args.manual_guidance,
                        "vis_spreads": args.vis_spreads,
                    },
                },
                environment={key: os.environ.get(key) for key in ROLLOUT_ENV}
                | {"LEROBOT_DEMO_NAME": os.environ.get("LEROBOT_DEMO_NAME", args.record)},
            )
            logging.info("Startup: policy/processor setup %.2fs", time.perf_counter() - setup_t)
            robot.connect()
            client.reset()
            start = time.perf_counter()
            first_action = True
            while time.perf_counter() - start < args.duration:
                step_start = time.perf_counter()
                observation = robot.get_observation()
                observation_ready = time.perf_counter()
                action = client.predict_action(observation, args.task, robot, raw_observation=True)
                robot.send_action(action)
                if first_action:
                    logging.info(
                        "First action: camera/state acquisition %.2fs, inference and command %.2fs",
                        observation_ready - step_start,
                        time.perf_counter() - observation_ready,
                    )
                    first_action = False
                precise_sleep(max(1 / args.fps - (time.perf_counter() - step_start), 0))
        finally:
            if robot.is_connected:
                robot.disconnect()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
