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
        with _defer_sigint(), h5py.File(self.path, "x") as file:
            file.attrs["schema_version"] = 2
            file.attrs["demo_name"] = demo_name
            file.attrs["created_at_utc"] = stamp
            file.attrs["metadata_json"] = json.dumps(metadata, default=str, sort_keys=True)
            file.attrs["action_space"] = "normalized policy output"
            file.attrs["image_space"] = "policy input, before model resizing; RGB BCHW [0, 1]"
            file.create_group("chunks")
            _write_tree(file.create_group("artifacts"), artifacts or {})
        logging.info("Recording inference trajectories to %s", self.path.resolve())

    def append(self, *, chunk_index, original, perturbed, selected_indices, observations, task, perturb_std,
               sampling_calls=(), raw_observation=None, perturbation_noise=None):
        """Persist one complete candidate set before guidance or action execution."""
        # Selection returns a tensor on the policy's device during CUDA inference.
        if hasattr(selected_indices, "detach"):
            selected_indices = selected_indices.detach().cpu().numpy()
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
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
            chunk.attrs["complete"] = True
            record_index = self._next_index
            self._next_index += 1
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
