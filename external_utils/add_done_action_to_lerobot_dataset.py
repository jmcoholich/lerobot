#!/usr/bin/env python

"""Append a terminal ``done`` dimension to an existing LeRobot action feature."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_feature_stats, get_feature_stats

try:
    from .add_annotations_to_lerobot_dataset import (
        find_data_files,
        load_json,
        write_json_atomic,
        write_table_atomic,
    )
except ImportError:
    from add_annotations_to_lerobot_dataset import (
        find_data_files,
        load_json,
        write_json_atomic,
        write_table_atomic,
    )


ACTION_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def find_episode_files(dataset_root: Path) -> list[Path]:
    episodes_root = dataset_root / "meta" / "episodes"
    paths = sorted(episodes_root.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata Parquet files found under {episodes_root}")
    return paths


def action_matrix(column: pa.ChunkedArray, path: Path) -> np.ndarray:
    values = column.combine_chunks().to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"{path} contains null action values")

    widths = {len(value) for value in values}
    if len(widths) != 1:
        raise ValueError(f"{path} contains inconsistent action widths: {sorted(widths)}")
    return np.asarray(values, dtype=np.float32)


def update_stat_dimension(
    values: list[float],
    done_value: float,
    original_width: int,
    context: str,
) -> tuple[list[float], bool]:
    if len(values) == original_width:
        return [*values, float(done_value)], True
    if len(values) != original_width + 1:
        raise ValueError(
            f"{context} has width {len(values)}, expected {original_width} or {original_width + 1}"
        )
    if not np.isclose(values[-1], done_value, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"{context} already has an inconsistent done statistic: {values[-1]} != {done_value}"
        )
    return values, False


def add_done_action(dataset_root: str | Path, *, dry_run: bool = False) -> dict[str, int | str]:
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"LeRobot statistics not found: {stats_path}")

    info = load_json(info_path)
    action_feature = info.get("features", {}).get("action")
    if action_feature is None:
        raise ValueError("Dataset metadata does not define an action feature")
    if action_feature.get("dtype") != "float32":
        raise ValueError(f"Expected float32 actions, got {action_feature.get('dtype')!r}")

    shape = action_feature.get("shape")
    names = action_feature.get("names")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(names, list):
        raise ValueError("Expected a one-dimensional, named action feature")
    if names and names[-1] == "done":
        target_width = shape[0]
        original_width = target_width - 1
        target_names = names
    else:
        if "done" in names:
            raise ValueError("Action name 'done' must be the final action dimension")
        original_width = shape[0]
        target_width = original_width + 1
        target_names = [*names, "done"]
    if len(names) != shape[0] or original_width < 1:
        raise ValueError(f"Action names and shape are inconsistent: names={names}, shape={shape}")

    data_paths = find_data_files(root)
    shards = []
    all_episode_indices = []
    all_frame_indices = []
    row_counts = []
    for path in data_paths:
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        if table.schema.field("action").type != pa.list_(pa.float32()):
            raise ValueError(f"{path} action column is not list<float32>")
        actions = action_matrix(table["action"], path)
        if actions.shape[1] not in (original_width, target_width):
            raise ValueError(
                f"{path} has action width {actions.shape[1]}, expected {original_width} or {target_width}"
            )
        episode_indices = table["episode_index"].combine_chunks().to_numpy()
        frame_indices = table["frame_index"].combine_chunks().to_numpy()
        shards.append((path, actions, actions.shape[1] == original_width))
        all_episode_indices.append(episode_indices)
        all_frame_indices.append(frame_indices)
        row_counts.append(table.num_rows)

    episode_indices = np.concatenate(all_episode_indices)
    frame_indices = np.concatenate(all_frame_indices)
    total_frames = len(episode_indices)
    if total_frames != info.get("total_frames"):
        raise ValueError(f"Found {total_frames} frames, metadata reports {info.get('total_frames')}")

    frame_indices_by_episode: dict[int, set[int]] = {}
    for episode_index, frame_index in zip(episode_indices, frame_indices, strict=True):
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        frames = frame_indices_by_episode.setdefault(episode_index, set())
        if frame_index in frames:
            raise ValueError(f"Episode {episode_index} contains duplicate frame_index {frame_index}")
        frames.add(frame_index)

    expected_episodes = set(range(info.get("total_episodes", -1)))
    if set(frame_indices_by_episode) != expected_episodes:
        raise ValueError("Data episode indices do not match metadata total_episodes")
    for episode_index, frames in frame_indices_by_episode.items():
        if frames != set(range(len(frames))):
            raise ValueError(
                f"Episode {episode_index} does not have contiguous frame indices starting at zero"
            )

    terminal_frames = {
        episode_index: max(frames) for episode_index, frames in frame_indices_by_episode.items()
    }
    done = np.asarray(
        [
            float(frame_index == terminal_frames[int(episode_index)])
            for episode_index, frame_index in zip(episode_indices, frame_indices, strict=True)
        ],
        dtype=np.float32,
    )
    if int(done.sum()) != len(expected_episodes):
        raise RuntimeError("Expected exactly one terminal frame per episode")

    updated_actions = []
    data_files_to_update = 0
    offset = 0
    for (path, actions, needs_update), row_count in zip(shards, row_counts, strict=True):
        shard_done = done[offset : offset + row_count]
        offset += row_count
        if needs_update:
            actions = np.concatenate([actions, shard_done[:, None]], axis=1, dtype=np.float32)
            data_files_to_update += 1
        elif not np.array_equal(actions[:, -1], shard_done):
            raise ValueError(f"{path} already has inconsistent done values")
        updated_actions.append(actions)

    done_stats_by_episode = {}
    for episode_index in sorted(expected_episodes):
        episode_done = done[episode_indices == episode_index, None]
        done_stats_by_episode[episode_index] = get_feature_stats(episode_done, axis=0, keepdims=False)

    episode_paths = find_episode_files(root)
    updated_episode_tables = []
    metadata_episode_indices = set()
    episode_files_to_update = 0
    for path in episode_paths:
        table = pq.read_table(path)
        required_columns = {
            "episode_index",
            "length",
            "stats/action/count",
            *(f"stats/action/{key}" for key in ACTION_STAT_KEYS),
        }
        missing = required_columns.difference(table.column_names)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        table_changed = False
        table_episode_indices = table["episode_index"].combine_chunks().to_pylist()
        table_lengths = table["length"].combine_chunks().to_pylist()
        for episode_index, length in zip(table_episode_indices, table_lengths, strict=True):
            episode_index = int(episode_index)
            if episode_index in metadata_episode_indices:
                raise ValueError(f"Duplicate episode metadata for episode {episode_index}")
            metadata_episode_indices.add(episode_index)
            if episode_index not in done_stats_by_episode:
                raise ValueError(f"Episode metadata contains unknown episode {episode_index}")
            if int(length) != len(frame_indices_by_episode[episode_index]):
                raise ValueError(f"Episode {episode_index} length does not match its data frames")

        counts = table["stats/action/count"].combine_chunks().to_pylist()
        for episode_index, length, count in zip(table_episode_indices, table_lengths, counts, strict=True):
            if count != [length]:
                raise ValueError(
                    f"Episode {episode_index} action-stat count {count} does not match length {length}"
                )

        updated_table = table
        for key in ACTION_STAT_KEYS:
            column_name = f"stats/action/{key}"
            values = table[column_name].combine_chunks().to_pylist()
            updated_values = []
            for episode_index, current_values in zip(table_episode_indices, values, strict=True):
                done_value = done_stats_by_episode[int(episode_index)][key][0]
                updated, changed = update_stat_dimension(
                    current_values,
                    done_value,
                    original_width,
                    f"episode {episode_index} {column_name}",
                )
                updated_values.append(updated)
                table_changed |= changed
            column_index = updated_table.schema.get_field_index(column_name)
            field = updated_table.schema.field(column_index)
            updated_table = updated_table.set_column(
                column_index,
                field,
                pa.array(updated_values, type=field.type),
            )

        updated_episode_tables.append((path, updated_table, table_changed))
        episode_files_to_update += int(table_changed)

    if metadata_episode_indices != expected_episodes:
        raise ValueError("Episode metadata indices do not match data episode indices")

    stats = load_json(stats_path)
    action_stats = stats.get("action")
    if action_stats is None:
        raise ValueError("Dataset statistics do not contain action statistics")
    if action_stats.get("count") != [total_frames]:
        raise ValueError(f"Global action-stat count does not match total frames: {action_stats.get('count')}")

    global_done_stats = aggregate_feature_stats(
        [done_stats_by_episode[episode_index] for episode_index in sorted(expected_episodes)]
    )
    stats_changed = False
    updated_action_stats = dict(action_stats)
    for key in ACTION_STAT_KEYS:
        if key not in action_stats:
            raise ValueError(f"Global action statistics are missing {key!r}")
        updated, changed = update_stat_dimension(
            action_stats[key],
            global_done_stats[key][0],
            original_width,
            f"global action {key}",
        )
        updated_action_stats[key] = updated
        stats_changed |= changed
    stats["action"] = updated_action_stats

    info_changed = shape != [target_width] or names != target_names
    action_feature["shape"] = [target_width]
    action_feature["names"] = target_names

    if not dry_run:
        for (path, _, needs_update), actions in zip(shards, updated_actions, strict=True):
            if actions.shape[1] != target_width:
                raise RuntimeError(f"Internal error: {path} was not migrated to width {target_width}")
            if needs_update:
                table = pq.read_table(path)
                column_index = table.schema.get_field_index("action")
                field = table.schema.field(column_index)
                table = table.set_column(column_index, field, pa.array(actions.tolist(), type=field.type))
                write_table_atomic(table, path)

        for path, table, changed in updated_episode_tables:
            if changed:
                write_table_atomic(table, path)

        if stats_changed:
            write_json_atomic(stats, stats_path)
        if info_changed:
            write_json_atomic(info, info_path)

        for path in data_paths:
            actions = action_matrix(pq.read_table(path, columns=["action"])["action"], path)
            if actions.shape[1] != target_width:
                raise RuntimeError(f"Verification failed: {path} action width is {actions.shape[1]}")

    return {
        "dataset": root.name,
        "files": len(data_paths),
        "updated_data_files": data_files_to_update,
        "updated_episode_files": episode_files_to_update,
        "frames": total_frames,
        "episodes": len(expected_episodes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root containing data/ and meta/",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = add_done_action(args.dataset_root, dry_run=args.dry_run)
    action = "Would migrate" if args.dry_run else "Migrated"
    print(
        f"{action} {result['dataset']}: {result['frames']} frames across {result['episodes']} episodes; "
        f"{result['updated_data_files']} data file(s) and "
        f"{result['updated_episode_files']} episode metadata file(s) required updates."
    )


if __name__ == "__main__":
    main()
