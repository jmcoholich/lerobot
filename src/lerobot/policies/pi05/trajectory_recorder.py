"""Incremental HDF5 logging for TACO inference trajectory candidates."""

import json
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np


def _rbf_kernel(*samples, gamma):
    """Build the pooled RBF kernel and resolve fixed or adaptive gamma."""
    samples = [np.asarray(value, dtype=np.float64) for value in samples]
    if any(value.ndim != 2 or not value.size for value in samples):
        raise ValueError("MMD inputs must be nonempty [samples, features] arrays")
    if any(value.shape[1] != samples[0].shape[1] for value in samples):
        raise ValueError("MMD inputs must be nonempty [samples, features] arrays with matching features")
    if any(not np.isfinite(value).all() for value in samples):
        raise ValueError("MMD inputs must be finite")
    z = np.concatenate(samples)
    distances = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    if isinstance(gamma, str):
        if gamma == "median":
            positive = distances[distances > 0]
            scale = 2 * np.median(positive) if positive.size else 0.0
        elif gamma == "max_eig":
            centered = z - z.mean(axis=0)
            # The sample Gram matrix shares the covariance's nonzero eigenvalues.
            scale = np.linalg.eigvalsh(centered @ centered.T)[-1] / (len(z) - 1) if len(z) > 1 else 0.0
        else:
            raise ValueError(f"Unsupported MMD gamma: {gamma}")
        gamma = 1.0 / scale if scale > 0 else 1.0
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("MMD gamma must be positive and finite")
    return np.exp(-gamma * distances), float(gamma)


def compute_mmd_rbf(current, previous, gamma="median"):
    """Return biased MMD squared and effective gamma between adjacent chunk overlaps.

    Each input has one candidate per row and a flattened, temporally aligned
    overlap per row. The RBF kernel is exp(-gamma * squared_distance).
    """
    kernel, gamma = _rbf_kernel(current, previous, gamma=gamma)
    num_current = len(current)
    within_current = kernel[:num_current, :num_current].mean()
    within_previous = kernel[num_current:, num_current:].mean()
    between_chunks = kernel[:num_current, num_current:].mean()
    mmd2 = max(within_current + within_previous - 2 * between_chunks, 0.0)
    return np.float64(mmd2), gamma


def compute_diversity_rbf(current, gamma):
    """Return 1-mean(k(current,current)) and effective gamma, including diagonals.

    Rows are candidate trajectories flattened over the measured horizon.
    """
    kernel, gamma = _rbf_kernel(current, gamma=gamma)
    return np.float64(1 - kernel.mean()), gamma


def as_numpy(value):
    """Snapshot a tensor without losing integer tokens/masks or aliasing CPU storage."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.array(value, copy=True)


def snapshot_processor(pipeline):
    """Embed the pipeline's standard JSON and safetensors files for offline loading."""
    with tempfile.TemporaryDirectory() as directory:
        pipeline.save_pretrained(directory, config_filename="processor.json", push_to_hub=False)
        return {
            path.name: np.frombuffer(path.read_bytes(), dtype=np.uint8)
            for path in Path(directory).iterdir() if path.is_file()
        }


def _write_tree(group, values):
    for name, value in values.items():
        if isinstance(value, dict):
            _write_tree(group.create_group(name), value)
        elif isinstance(value, str):
            group.attrs[name] = value
        elif value is not None:
            value = as_numpy(value)
            group.create_dataset(name, data=value, compression="lzf" if value.ndim else None)


@contextmanager
def _defer_sigint():
    """Finish closing the HDF5 file before delivering a pending Ctrl+C."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    interrupted = False

    def defer(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if interrupted:
            signal.raise_signal(signal.SIGINT)


class TrajectoryRecorder:
    """Close the file after every chunk so normal exit and SIGINT need no final save.

    Actions are normalized policy outputs, shaped [samples, n_action_steps, action_dim].
    Observations retain their policy-input layout and values (images are normally
    RGB float32 [batch, channels, height, width] in [0, 1], before model resizing).
    """

    def __init__(self, metadata: dict, output_dir: str | Path | None = None, demo_name=None, artifacts=None):
        import h5py

        self._h5py = h5py
        output_dir = Path(output_dir or os.environ.get("LEROBOT_TRAJECTORY_DIR", "outputs/inference_trajectories"))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        demo_name = demo_name or os.environ.get("LEROBOT_DEMO_NAME")
        if not demo_name:
            demo_name = output_dir.name.removeprefix("demonstration_") if output_dir.name.startswith("demonstration_") else "last_recording"
        if demo_name in (".", "..") or Path(demo_name).name != demo_name:
            raise ValueError("The trajectory demo name must be a filename, without directory separators")
        self.path = output_dir / f"trajectories_{demo_name}.h5"
        self._next_index = 0
        self._mmd_gamma = metadata.get("policy_config", {}).get("mmd_gamma", "median")
        self._diversity_gamma = metadata.get("policy_config", {}).get("diversity_gamma", 0.06302211495246096)
        self.reset()
        with _defer_sigint(), h5py.File(self.path, "w") as file:
            file.attrs["schema_version"] = 2
            file.attrs["demo_name"] = demo_name
            file.attrs["created_at_utc"] = stamp
            file.attrs["metadata_json"] = json.dumps(metadata, default=str, sort_keys=True)
            file.attrs["action_space"] = "normalized policy output"
            file.attrs["image_space"] = "policy input, before model resizing; RGB BCHW [0, 1]"
            file.create_group("chunks")
            _write_tree(file.create_group("artifacts"), artifacts or {})
        logging.info("Recording inference trajectories to %s", self.path.resolve())

    def reset(self):
        """Discard the overlap when an episode resets, including a partially executed chunk."""
        self._previous_overlap = None
        self._previous_record_index = -1

    def append(self, *, chunk_index, original, perturbed, selected_indices, observations, task, perturb_std,
               sampling_calls=(), raw_observation=None, perturbation_noise=None):
        """Persist one complete candidate set before guidance or action execution."""
        # Selection returns a tensor on the policy's device during CUDA inference.
        if hasattr(selected_indices, "detach"):
            selected_indices = selected_indices.detach().cpu().numpy()
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        # Use all unperturbed candidates, excluding padding and the final Franka gripper dimension.
        full_actions = sampling_calls[0]["full_output"] if sampling_calls else original
        full_actions = np.asarray(full_actions)[..., :original.shape[-1] - 1]
        exec_horizon = original.shape[1]
        overlap_steps = full_actions.shape[1] - exec_horizon
        diversity_steps = overlap_steps if overlap_steps > 0 else full_actions.shape[1]
        current = full_actions[:, :diversity_steps].reshape(len(full_actions), -1)
        previous = self._previous_overlap if overlap_steps > 0 else None
        score, gamma = np.nan, np.nan
        if previous is not None:
            score, gamma = compute_mmd_rbf(current, previous, self._mmd_gamma)
        diversity, diversity_gamma = compute_diversity_rbf(current, self._diversity_gamma)
        with _defer_sigint(), self._h5py.File(self.path, "a") as file:
            chunk = file["chunks"].create_group(f"{self._next_index:06d}")
            chunk.attrs["complete"] = False
            chunk.attrs["chunk_index"] = chunk_index
            chunk.attrs["timestamp_unix"] = time.time()
            chunk.attrs["task_json"] = json.dumps(task, default=str)
            chunk.attrs["perturb_std"] = perturb_std
            chunk.attrs["execution_ready"] = False
            for name, value in {
                "original_actions": original,
                "perturbed_actions": perturbed,
                "selected_actions": perturbed[selected_indices],
                "selected_indices": selected_indices,
            }.items():
                chunk.create_dataset(name, data=value, compression="lzf")
            images = chunk.create_group("observations")
            for name, value in observations.items():
                images.create_dataset(name, data=value, compression="lzf")
            _write_tree(chunk.create_group("raw_observation"), raw_observation or {})
            if perturbation_noise is not None:
                chunk.create_dataset("perturbation_noise", data=as_numpy(perturbation_noise), compression="lzf")
            sampling = chunk.create_group("sampling")
            for index, call in enumerate(sampling_calls):
                _write_tree(sampling.create_group(f"{index:06d}"), call)
            _write_tree(chunk.create_group("temporal_mmd"), {
                "mmd2": score,
                "gamma": gamma,
                "gamma_setting": str(self._mmd_gamma),
                "kernel": "exp(-gamma * squared_distance)",
                "estimator": "biased MMD squared over flattened normalized action sequences",
                "source": "all unperturbed candidates, sampling/000000/full_output, without padding or the final gripper dimension",
                "overlap_steps": overlap_steps,
                "num_samples": len(full_actions),
                "action_dim": full_actions.shape[-1],
                "previous_record_index": self._previous_record_index,
            })
            _write_tree(chunk.create_group("candidate_diversity"), {
                "score": diversity,
                "gamma": diversity_gamma,
                "horizon": diversity_steps,
                "num_samples": len(full_actions),
                "action_dim": full_actions.shape[-1],
                "estimator": "1-mean(k(current,current)), including diagonals; higher means more diverse",
                "source": "all unperturbed current candidates, overlap prefix or full horizon if no overlap, without padding or the final gripper dimension",
            })
            chunk.attrs["complete"] = True
            record_index = self._next_index
            self._next_index += 1
            self._previous_overlap = (
                full_actions[:, exec_horizon:].reshape(len(full_actions), -1).copy()
                if overlap_steps > 0 else None
            )
            self._previous_record_index = record_index
        if np.isfinite(score):
            logging.info("Chunk %s temporal MMD²=%.6f (gamma=%.6g, overlap=%s)",
                         chunk_index, score, gamma, overlap_steps)
        return record_index

    def record_execution(self, record_index, *, sampling_calls, actions, source):
        """Record guidance sampling and the normalized action chunk queued for execution."""
        with _defer_sigint(), self._h5py.File(self.path, "a") as file:
            chunk = file[f"chunks/{record_index:06d}"]
            sampling = chunk["sampling"]
            offset = len(sampling)
            for index, call in enumerate(sampling_calls, start=offset):
                _write_tree(sampling.create_group(f"{index:06d}"), call)
            chunk.create_dataset("queued_actions", data=as_numpy(actions), compression="lzf")
            chunk.attrs["execution_source"] = source
            chunk.attrs["execution_ready"] = True
