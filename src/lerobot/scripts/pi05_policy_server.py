"""Persistent, single-client π0.5 inference over a local Unix socket.

Run with ``python -m lerobot.scripts.pi05_policy_server``. The recorder sends
the resolved policy config on connection; no checkpoint is loaded at startup.
"""

import argparse
import gc
import logging
import os
import pickle
from contextlib import suppress
from dataclasses import asdict
from multiprocessing.connection import Client, Listener
from types import SimpleNamespace

DEFAULT_SOCKET = f"/tmp/lerobot-pi05-{os.getuid()}.sock"
ROLLOUT_ENV = ("LEROBOT_DEMO_NAME", "LEROBOT_TRAJECTORY_DIR")


class RemotePolicyClient:
    def __init__(self, address):
        try:
            self.connection = Client(address, family="AF_UNIX")
        except OSError as error:
            raise ConnectionError(
                f"Cannot connect to {address}. Start bash pi_05_policy_server.bash before running inference"
            ) from error

    def request(self, operation, **kwargs):
        self.connection.send((operation, kwargs))
        status, result = self.connection.recv()
        if status == "stop":
            raise SystemExit(result)
        if status == "error":
            raise RuntimeError(f"Policy server: {result}")
        return result

    def configure(self, cfg, dataset_meta):
        self.request(
            "configure",
            policy_cfg=cfg.policy,
            dataset_meta=SimpleNamespace(features=dataset_meta.features, stats=dataset_meta.stats),
            rename_map=cfg.dataset.rename_map,
            rollout_config=asdict(cfg),
            environment={key: os.environ.get(key) for key in ROLLOUT_ENV},
        )

    def reset(self):
        self.request("reset")

    def predict_action(self, observation, task, robot, *, raw_observation=False):
        # TACO only reads these Franka poses; hardware stays in the recorder process.
        robot_state = None
        if robot.robot_type == "franka":
            robot_state = SimpleNamespace(debug=robot.debug)
            if not robot.debug:
                interface = robot.operator.robot_interface
                robot_state.operator = SimpleNamespace(
                    robot_interface=SimpleNamespace(
                        last_eef_pose=interface.last_eef_pose,
                        last_eef_quat_and_pos=interface.last_eef_quat_and_pos,
                    )
                )
        action = self.request(
            "predict_robot" if raw_observation else "predict",
            observation=observation,
            task=task,
            robot_type=robot.robot_type,
            robot=robot_state,
        )
        if raw_observation:
            return action
        import torch

        return torch.from_numpy(action)

    def close(self):
        self.connection.close()


class PolicySession:
    def __init__(self):
        self.policy = self.preprocessor = self.postprocessor = None
        self.cache_key = None

    def configure_robot(self, policy_path, policy_overrides, robot_features, rollout_config, environment):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.utils import hw_to_dataset_features
        from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401

        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=policy_overrides)
        policy_cfg.pretrained_path = policy_path
        features = {
            **hw_to_dataset_features(robot_features["action"], "action"),
            **hw_to_dataset_features(robot_features["observation"], "observation"),
        }
        rollout_config["policy"] = asdict(policy_cfg)
        self.configure(policy_cfg, SimpleNamespace(features=features, stats={}), {}, rollout_config, environment)
        self.robot_features = features

    def predict_robot(self, observation, **kwargs):
        from lerobot.datasets.utils import build_dataset_frame

        observation = build_dataset_frame(self.robot_features, observation, "observation")
        action = self.predict(observation=observation, **kwargs).squeeze(0)
        return dict(zip(self.robot_features["action"]["names"], action.tolist(), strict=True))

    def configure(self, policy_cfg, dataset_meta, rename_map, rollout_config, environment):
        import torch

        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.processor.rename_processor import rename_stats

        if policy_cfg.type != "pi05" or not policy_cfg.pretrained_path:
            raise ValueError("The server requires a pi05 policy with --policy.path")
        # Capture before make_policy fills feature fields in the config.
        key = pickle.dumps((asdict(policy_cfg), dataset_meta.features, dataset_meta.stats, rename_map))
        if key != self.cache_key:
            self.policy = self.preprocessor = self.postprocessor = None
            self.cache_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Loading policy: %s", policy_cfg.pretrained_path)
            self.policy = make_policy(policy_cfg, ds_meta=dataset_meta)
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=policy_cfg.pretrained_path,
                dataset_stats=rename_stats(dataset_meta.stats, rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": policy_cfg.device},
                    "rename_observations_processor": {"rename_map": rename_map},
                },
            )
            self.cache_key = key
        else:
            logging.info("Reusing cached policy: %s", policy_cfg.pretrained_path)
        for name in ROLLOUT_ENV:
            if environment.get(name) is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = environment[name]
        self.policy.reset_rollout()
        self.policy.configure_replay_recording(rollout_config)
        self.reset()

    def reset(self):
        for component in (self.policy, self.preprocessor, self.postprocessor):
            component.reset()

    def predict(self, **kwargs):
        from lerobot.utils.control_utils import predict_action
        from lerobot.utils.utils import get_safe_torch_device

        action = predict_action(
            **kwargs,
            policy=self.policy,
            device=get_safe_torch_device(self.policy.config.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.policy.config.use_amp,
        )
        # Send arrays by value, without PyTorch's process-lifetime tensor handles.
        return action.detach().cpu().float().numpy()


def serve_connection(connection, session):
    configured = False
    while True:
        try:
            operation, kwargs = connection.recv()
        except EOFError:
            return
        try:
            if operation in ("configure", "configure_robot"):
                configured = False
                getattr(session, operation)(**kwargs)
                configured = True
                result = None
            elif not configured:
                raise RuntimeError("Configure the policy before requesting inference")
            elif operation == "reset":
                result = session.reset()
            elif operation in ("predict", "predict_robot"):
                result = getattr(session, operation)(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            response = ("ok", result)
        except SystemExit as error:
            # TACO's step limit ends the rollout, not the persistent server.
            connection.send(("stop", error.code))
            return
        except Exception as error:
            logging.exception("Policy request failed")
            connection.send(("error", f"{type(error).__name__}: {error}"))
            return
        connection.send(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    session = PolicySession()
    # Only this Unix user may connect; the protocol contains Python objects.
    previous_umask = os.umask(0o077)
    try:
        listener = Listener(args.socket, family="AF_UNIX")
    finally:
        os.umask(previous_umask)
    with listener:
        logging.info("Empty policy server listening on %s", args.socket)
        while True:
            with listener.accept() as connection:
                try:
                    serve_connection(connection, session)
                except (EOFError, ConnectionError, OSError):
                    logging.info("Rollout disconnected; keeping policy loaded")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        main()
