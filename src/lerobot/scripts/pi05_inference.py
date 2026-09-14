"""Franka policy-server client without LeRobot dataset recording or model imports."""

import argparse
import logging
import os
import time
from contextlib import closing
from dataclasses import asdict

from lerobot.robots.franka import FrankaConfig, FrankaRobot
from lerobot.scripts.pi05_policy_server import DEFAULT_SOCKET, ROLLOUT_ENV, RemotePolicyClient
from lerobot.scripts.pi05_timing import RolloutTiming
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
    parser.add_argument("--ensemble_request_mode", choices=["parallel", "serial"], default="parallel",
                        help="How to send PIVOT and primitive requests in ensemble mode")
    parser.add_argument("--vlm_server_url", required=True,
                        help="Base URL of the OpenAI-compatible VLM service for interventions")
    parser.add_argument("--vlm_model_name", default="Qwen/Qwen2.5-VL-72B-Instruct",
                        help="Model identifier served by the VLM service")
    parser.add_argument("--manual_guidance", choices=["true", "false"], default="false")
    parser.add_argument("--vis_spreads", choices=["true", "false"], default="false")
    args, overrides = parser.parse_known_args(argv)
    if not args.vlm_server_url.strip():
        parser.error("--vlm_server_url must not be empty")
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
        timing = None
        end_reason = "duration_limit"
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
                        "ensemble_request_mode": args.ensemble_request_mode,
                        "vlm_server_url": args.vlm_server_url,
                        "vlm_model_name": args.vlm_model_name,
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
            timing = RolloutTiming(client)
            start = timing.start
            first_action = True
            while time.perf_counter() - start < args.duration:
                step_start = timing.begin_step()
                observation = robot.get_observation()
                observation_ready = time.perf_counter()
                action = client.predict_action(observation, args.task, robot, raw_observation=True, with_timing=True)
                timing.prediction_done(client.last_timing_info, observation_ready - step_start,
                                       time.perf_counter() - observation_ready)
                command_start = timing.begin_command()
                robot.send_action(action)
                timing.command_done(command_start)
                if first_action:
                    logging.info(
                        "First action: camera/state acquisition %.2fs, inference and command %.2fs",
                        observation_ready - step_start,
                        time.perf_counter() - observation_ready,
                    )
                    first_action = False
                precise_sleep(max(1 / args.fps - (time.perf_counter() - step_start), 0))
                timing.end_step(client.last_timing_info["chunk_complete"])
        except BaseException as error:
            end_reason = type(error).__name__
            raise
        finally:
            try:
                if timing is not None:
                    timing.report(finalized=True, end_reason=end_reason)
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
