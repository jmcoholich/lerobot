"""Render the full predicted horizon of the final five trajectories in a six-column PNG.

Run in the inference environment:
    python scripts/visualize_pi05_trajectories.py /path/to/trajectories_demo.h5
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import cv2
import h5py
import numpy as np
from safetensors.numpy import load as load_safetensors


FRONT_IMAGE = "observations/observation.images.camera_front"
COLUMNS = 6
GAP = 8
LABEL_HEIGHT = 28


@contextmanager
def isolated_import_directory(repo):
    # The existing model module clears ./vlm_io on import. Isolate that side effect.
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pi05-visualization-") as directory:
        (Path(directory) / "src").symlink_to(repo / "src", target_is_directory=True)
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous)


def load_renderer():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo.parent / "openteach"))
    with isolated_import_directory(repo):
        from lerobot.policies.pi05.modelling_pi05_taco import visualize_trajectories_on_camera
    return visualize_trajectories_on_camera


def load_plot_postprocessor(file):
    """Read the action ranges the existing visualization uses, entirely on CPU."""
    if "artifacts/postprocessor/processor.json" not in file:
        raise ValueError("This log has no saved postprocessor; use a schema-2 trajectory recording.")
    artifacts = file["artifacts/postprocessor"]
    config = json.loads(artifacts["processor.json"][()].tobytes())
    for step in config["steps"]:
        state_file = step.get("state_file")
        if state_file is None:
            continue
        stats = load_safetensors(artifacts[state_file][()].tobytes())
        if "action.min" in stats and "action.max" in stats:
            return SimpleNamespace(steps=[SimpleNamespace(stats={"action": {
                "min": stats["action.min"], "max": stats["action.max"],
            }})])
    raise ValueError("The saved postprocessor has no action.min/action.max ranges for projection.")


def full_selected_trajectories(chunk):
    """Extend the recorded selected prefix with its original, unperturbed predictions."""
    if "sampling/000000/full_output" not in chunk:
        raise ValueError(f"{chunk.name}: no full sampler output was recorded.")
    selected = chunk["selected_actions"][()]
    indices = chunk["selected_indices"][()]
    full = chunk["sampling/000000/full_output"][()]
    if selected.ndim != 3 or selected.shape[0] != 5 or selected.shape[2] < 3:
        raise ValueError(f"{chunk.name}: expected five XYZ trajectories; got {selected.shape}.")
    if full.ndim != 3 or full.shape[1] < selected.shape[1] or full.shape[2] < selected.shape[2]:
        raise ValueError(f"{chunk.name}: full sampler output is incompatible with selected trajectories.")
    actions = full[indices, :, :selected.shape[2]].copy()
    # Noise was only applied to the execution horizon. Preserve those exact
    # selected points; the remainder comes from the same original sample IDs.
    actions[:, :selected.shape[1]] = selected
    return actions


def combine_tiles(tiles):
    """Lay out equally-sized BGR camera tiles, six per row, without scaling them."""
    if not tiles:
        raise ValueError("No complete trajectory chunks with front-camera observations were found.")
    height, width, channels = tiles[0][1].shape
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    grid = np.full((rows * (height + LABEL_HEIGHT) + (rows + 1) * GAP,
                    COLUMNS * width + (COLUMNS + 1) * GAP, channels), 24, dtype=np.uint8)
    for index, (label, tile) in enumerate(tiles):
        if tile.shape != (height, width, channels):
            raise ValueError("Front-camera tile dimensions differ between chunks.")
        row, column = divmod(index, COLUMNS)
        x = GAP + column * (width + GAP)
        y = GAP + row * (height + LABEL_HEIGHT + GAP)
        cv2.putText(grid, label, (x + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        grid[y + LABEL_HEIGHT:y + LABEL_HEIGHT + height, x:x + width] = tile
    return grid


def recorded_run_time(file):
    """Format the saved recording time, never the PNG creation or file modification time."""
    stamp = file.attrs.get("created_at_utc")
    if stamp:
        if isinstance(stamp, bytes):
            stamp = stamp.decode("utf-8")
        recorded = datetime.strptime(stamp, "%Y%m%dT%H%M%S_%fZ").replace(tzinfo=timezone.utc)
    else:
        timestamps = [chunk.attrs["timestamp_unix"] for chunk in file["chunks"].values()
                      if "timestamp_unix" in chunk.attrs]
        if not timestamps:
            return "not recorded"
        recorded = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
    return recorded.astimezone(ZoneInfo("America/New_York")).strftime("%B %d, %Y at %I:%M:%S %p %Z")


def add_metadata_header(grid, metadata, run_time="not recorded"):
    """Render recorded run settings above the grid, wrapping long checkpoint paths."""
    policy = metadata.get("policy_config") or (metadata.get("rollout_config") or {}).get("policy") or {}
    checkpoint = metadata.get("checkpoint_file") or policy.get("pretrained_path") or "not recorded"
    intervention = metadata.get("interventions", "not recorded")
    if intervention is False or intervention is None or str(intervention).lower() in ("false", "none", "no intervention"):
        intervention = "no intervention"
    else:
        intervention = str(intervention).lower()
    if metadata.get("manual_guidance"):
        intervention = "manual guidance"
    elif metadata.get("vis_spreads"):
        intervention = "no intervention (visualization only)"
    text_lines = [
        f"Recorded: {run_time}",
        f"Checkpoint: {checkpoint}",
        f"Intervention: {intervention}",
        f"n_action_steps: {policy.get('n_action_steps', 'not recorded')}",
    ]
    margin, line_height, scale = 16, 30, 0.65
    max_width = grid.shape[1] - 2 * margin
    wrapped = []
    for text in text_lines:
        line = ""
        for character in text:
            candidate = line + character
            if line and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_width:
                wrapped.append(line)
                line = character
            else:
                line = candidate
        wrapped.append(line)
    header = np.full((2 * margin + len(wrapped) * line_height, grid.shape[1], 3), 24, dtype=np.uint8)
    for index, line in enumerate(wrapped):
        cv2.putText(header, line, (margin, margin + 21 + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, grid], axis=0)


def create_visualization(log_path, output_path=None):
    import torch

    log_path = Path(log_path).resolve()
    output_path = Path(output_path).resolve() if output_path else log_path.with_suffix(".png")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Use a .png output filename for the combined image.")
    tiles = []
    with h5py.File(log_path, "r") as file:
        metadata = json.loads(file.attrs.get("metadata_json", "{}"))
        run_time = recorded_run_time(file)
        postprocessor = load_plot_postprocessor(file)
        render = load_renderer()
        for key in sorted(file["chunks"], key=int):
            chunk = file["chunks"][key]
            if not chunk.attrs.get("complete", False):
                print(f"Skipping incomplete chunk {key}")
                continue
            if FRONT_IMAGE not in chunk or "selected_actions" not in chunk:
                raise ValueError(f"Chunk {key} is missing its front image or selected trajectories.")
            image = chunk[FRONT_IMAGE][()]
            actions = full_selected_trajectories(chunk)
            if image.shape != (1, 3, 360, 640):
                raise ValueError(f"Chunk {key}: expected a [1, 3, 360, 640] image for the current camera calibration; got {image.shape}.")
            if actions.ndim != 3 or actions.shape[0] != 5 or actions.shape[2] < 3:
                raise ValueError(f"Chunk {key}: expected five XYZ trajectories; got {actions.shape}.")
            front, _ = render(
                {"observation.images.camera_front": torch.from_numpy(image)},
                actions.copy(), robot=None, actions_are_normalized=True,
                postprocessor=postprocessor, save_imgs=False, trajectory_stride=1,
            )
            tiles.append((f"Chunk {chunk.attrs.get('chunk_index', int(key))} | {actions.shape[1]} steps", front))
    grid = add_metadata_header(combine_tiles(tiles), metadata, run_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), grid):
        raise OSError(f"Could not write {output_path}")
    print(f"Saved {len(tiles)} chunks in {COLUMNS} columns: {output_path} ({grid.shape[1]} x {grid.shape[0]})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Trajectory HDF5 file")
    parser.add_argument("--output", "-o", type=Path, help="Output PNG (default: same name and folder as the HDF5 file)")
    args = parser.parse_args()
    try:
        create_visualization(args.log, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
