"""Verify exact GPU replay of a trajectory log without instantiating a robot.

Run with the same environment used for inference:
    python scripts/verify_pi05_trajectory_replay.py LOG.h5 --report REPORT.json
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

import draccus
import h5py
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def isolated_working_directory(repo):
    # The model module clears ./vlm_io at import. Isolate that existing side effect
    # while keeping its relative prompt-template reads available through ./src.
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pi05-replay-") as directory:
        (Path(directory) / "src").symlink_to(repo / "src", target_is_directory=True)
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous)


def comparison(actual, expected):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    expected = np.asarray(expected)
    same_layout = actual.shape == expected.shape and actual.dtype == expected.dtype
    bitwise_equal = same_layout and actual.tobytes() == expected.tobytes()
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "bitwise_equal": bitwise_equal,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "elements": int(actual.size),
        "unequal_elements": int(np.count_nonzero(actual != expected)),
        "max_abs_error": float(delta.max()) if delta.size else 0.0,
    }


@torch.inference_mode()
def verify(log_path):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modelling_pi05_taco import PI05PolicyTaco, select_representative_trajectories
    from lerobot.policies.pi05.trajectory_replay import load_recorded_processor, replay_sampling_call

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the sandbox with GPU access")
    with h5py.File(log_path, "r") as file:
        metadata = json.loads(file.attrs["metadata_json"])
        config = draccus.decode(PI05Config, metadata["policy_config"])
        checkpoint = Path(metadata["checkpoint_file"])
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint SHA-256 does not match the recording")
        source = Path(sys.modules[PI05PolicyTaco.__module__].__file__)
        source_matches = source.read_bytes() == file[f"artifacts/source/{source.name}"][()].tobytes()
        runtime = {
            "torch_version": torch.__version__, "transformers_version": version("transformers"),
            "cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(),
        }
        print(json.dumps({"checkpoint_hash_matches": True, "source_matches": source_matches,
                          "runtime": runtime}, indent=2), flush=True)
        torch.backends.cuda.matmul.allow_tf32 = metadata["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = metadata["cudnn_allow_tf32"]
        torch.use_deterministic_algorithms(metadata["deterministic_algorithms"])
        policy = PI05PolicyTaco.from_pretrained(str(checkpoint.parent), config=config, local_files_only=True)
        policy.eval()
        preprocessor = load_recorded_processor(log_path, "preprocessor")
        postprocessor = load_recorded_processor(log_path, "postprocessor")
        # Deliberately choose fresh RNG states; archived initial noise must suffice.
        torch.manual_seed(20260910)
        torch.cuda.manual_seed_all(20260910)
        report = {
            "log": str(log_path), "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "checkpoint_hash_matches": True, "source_matches": source_matches,
            "runtime": runtime, "runtime_matches": {key: value == metadata[key] for key, value in runtime.items()},
            "chunks": {},
        }
        for key, chunk in file["chunks"].items():
            if not chunk.attrs["complete"] or not chunk.attrs["execution_ready"]:
                raise RuntimeError(f"Chunk {key} is incomplete")
            checks = {}
            outputs = []
            for call_key, call in chunk["sampling"].items():
                output = replay_sampling_call(policy.model, call, "cuda")
                checks[f"sampling/{call_key}/full_output"] = comparison(output, call["full_output"][()])
                outputs.append(output)
            horizon, action_dim = chunk["original_actions"].shape[1:]
            original = outputs[0][:, :horizon, :action_dim]
            perturbed = original.clone()
            noise = torch.from_numpy(chunk["perturbation_noise"][()]).cuda()
            perturbed[:, 1:, :3] += noise * chunk.attrs["perturb_std"]
            indices = select_representative_trajectories(perturbed, num_trajectories=5)
            selected = perturbed[indices]
            source = chunk.attrs["execution_source"]
            if source == "original_sample_0":
                queued = original[:1]
            elif source == "selected_sample_0":
                queued = selected[:1]
            elif source == "last_sampling_call":
                queued = outputs[-1][:, :horizon, :action_dim]
            else:
                raise ValueError(source)
            queued = queued[:, :chunk["queued_actions"].shape[1]]
            for name, value in (("original_actions", original), ("perturbed_actions", perturbed),
                                ("selected_indices", indices), ("selected_actions", selected), ("queued_actions", queued)):
                checks[name] = comparison(value, chunk[name][()])

            raw = {name: torch.from_numpy(value[()]).cuda() for name, value in chunk["observations"].items()
                   if name.startswith("observation.images.")}
            raw["observation.state"] = torch.from_numpy(chunk["raw_observation/observation.state"][()]).cuda()
            raw["task"] = json.loads(chunk["raw_observation"].attrs["task_json"])
            prepared = preprocessor(raw)
            for name, expected in chunk["observations"].items():
                checks[f"preprocessor/{name}"] = comparison(prepared[name], expected[()])
            images, masks = policy._preprocess_images(prepared)
            for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
                checks[f"model_image/{index}"] = comparison(image, chunk[f"sampling/000000/inputs/images/{index}"][()])
                checks[f"model_image_mask/{index}"] = comparison(mask, chunk[f"sampling/000000/inputs/image_masks/{index}"][()])
            saved_queue = torch.from_numpy(chunk["queued_actions"][()]).cuda()
            actual_robot_actions = torch.stack([postprocessor(queued[:, step].clone()) for step in range(horizon)], dim=1)
            expected_robot_actions = torch.stack([postprocessor(saved_queue[:, step].clone()) for step in range(horizon)], dim=1)
            checks["postprocessed_queued_actions"] = comparison(actual_robot_actions, expected_robot_actions.cpu().numpy())
            report["chunks"][key] = checks
            failed = {name: check for name, check in checks.items() if not check["bitwise_equal"]}
            print(f"Chunk {key}: {len(checks)} checks; " + (f"mismatches: {json.dumps(failed)}" if failed else "all bitwise identical"), flush=True)
        report["all_bitwise_equal"] = all(check["bitwise_equal"] for checks in report["chunks"].values() for check in checks.values())
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    log = args.log.resolve()
    report_path = args.report.resolve()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, "/home/jeremiah/openteach")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with isolated_working_directory(repo):
        report = verify(log)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {report_path}", flush=True)
    raise SystemExit(0 if report["all_bitwise_equal"] else 1)


if __name__ == "__main__":
    main()
