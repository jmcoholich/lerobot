#!/usr/bin/env python

"""Add annotation-derived rewards and discounted returns to LeRobot datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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


DEFAULT_DATASET_ROOTS = (
    Path("/data3/lerobot_data/plug5_offline_rl_dataset"),
    Path("/data3/lerobot_data/walle_skywalker_testset"),
)
GAMMAS = (0.999, 0.995, 0.99, 0.95, 0.9)
ANNOTATION_REWARDS = {
    "failure": -1.0,
    "partial success": 0.5,
    "partial success to full success": 0.5,
    "plug picked": 0.25,
    "success": 1.0,
}
STAT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
FLOAT_FEATURE = {"dtype": "float32", "shape": [1], "names": None}


def return_column(gamma: float) -> str:
    return f"annotation_return_gamma_{gamma}"


def output_columns() -> list[str]:
    return ["annotation_reward", *(return_column(gamma) for gamma in GAMMAS)]


def annotation_rewards(annotations: list[str | None]) -> tuple[np.ndarray, Counter[str]]:
    rewards = np.zeros(len(annotations), dtype=np.float32)
    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()

    for index, raw_annotation in enumerate(annotations):
        annotation = "" if raw_annotation is None else str(raw_annotation).strip()
        if not annotation:
            continue
        if annotation not in ANNOTATION_REWARDS:
            unknown[annotation] += 1
            continue
        rewards[index] = ANNOTATION_REWARDS[annotation]
        counts[annotation] += 1

    if unknown:
        details = ", ".join(f"{value!r} ({count})" for value, count in sorted(unknown.items()))
        raise ValueError(f"Unknown non-empty annotations: {details}")

    return rewards, counts


def discounted_returns(
    rewards: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Compute returns independently per episode and restore the original row order."""
    if not (len(rewards) == len(episode_indices) == len(frame_indices)):
        raise ValueError("Reward, episode_index, and frame_index lengths do not match")

    returns = np.empty((len(rewards), len(GAMMAS)), dtype=np.float32)
    gammas = np.asarray(GAMMAS, dtype=np.float32)

    for episode_index in np.unique(episode_indices):
        row_indices = np.flatnonzero(episode_indices == episode_index)
        order = np.argsort(frame_indices[row_indices], kind="stable")
        sorted_rows = row_indices[order]
        sorted_frames = frame_indices[sorted_rows]
        if len(np.unique(sorted_frames)) != len(sorted_frames):
            raise ValueError(f"Episode {episode_index} contains duplicate frame_index values")

        running = np.zeros(len(GAMMAS), dtype=np.float32)
        for row_index in sorted_rows[::-1]:
            running = rewards[row_index] + gammas * running
            returns[row_index] = running

    return returns


def scalar_feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(values):
        raise ValueError("Cannot compute statistics for an empty feature")
    stats: dict[str, list[float] | list[int]] = {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values, dtype=np.float64))],
        "std": [float(np.std(values, dtype=np.float64))],
        "count": [int(len(values))],
    }
    for quantile in STAT_QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = [float(np.quantile(values, quantile))]
    return stats


def add_annotation_rewards(dataset_root: str | Path, *, dry_run: bool = False) -> dict[str, object]:
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")

    info = load_json(info_path)
    for column in output_columns():
        existing = info.get("features", {}).get(column)
        if existing is not None and existing.get("dtype") != "float32":
            raise ValueError(
                f"Metadata feature {column!r} has dtype {existing.get('dtype')!r}, expected 'float32'"
            )

    data_paths = find_data_files(root)
    tables: list[pa.Table] = []
    annotations: list[str | None] = []
    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    row_counts: list[int] = []

    for path in data_paths:
        schema_names = set(pq.read_schema(path).names)
        required = {"annotation", "episode_index", "frame_index"}
        missing = required - schema_names
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        table = pq.read_table(path)
        tables.append(table)
        row_counts.append(table.num_rows)
        annotations.extend(table["annotation"].combine_chunks().to_pylist())
        episode_parts.append(table["episode_index"].combine_chunks().to_numpy())
        frame_parts.append(table["frame_index"].combine_chunks().to_numpy())

    rewards, annotation_counts = annotation_rewards(annotations)
    episode_indices = np.concatenate(episode_parts)
    frame_indices = np.concatenate(frame_parts)
    returns = discounted_returns(rewards, episode_indices, frame_indices)
    columns = output_columns()
    values_by_column = {
        "annotation_reward": rewards,
        **{return_column(gamma): returns[:, index] for index, gamma in enumerate(GAMMAS)},
    }

    if not dry_run:
        offset = 0
        for path, table, row_count in zip(data_paths, tables, row_counts, strict=True):
            updated = table
            for column in columns:
                values = pa.array(values_by_column[column][offset : offset + row_count], type=pa.float32())
                column_index = updated.schema.get_field_index(column)
                if column_index >= 0:
                    updated = updated.set_column(column_index, column, values)
                else:
                    updated = updated.append_column(column, values)
            write_table_atomic(updated, path)
            offset += row_count

        features = info.setdefault("features", {})
        for column in columns:
            features[column] = FLOAT_FEATURE.copy()
        write_json_atomic(info, info_path)

        stats_path = root / "meta" / "stats.json"
        stats = load_json(stats_path) if stats_path.is_file() else {}
        for column, values in values_by_column.items():
            stats[column] = scalar_feature_stats(values)
        write_json_atomic(stats, stats_path)

    return {
        "dataset": root.name,
        "files": len(data_paths),
        "frames": len(rewards),
        "annotated_frames": int(sum(annotation_counts.values())),
        "annotation_counts": dict(sorted(annotation_counts.items())),
        "columns": columns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASET_ROOTS),
        help="Dataset roots; defaults to plug5_offline_rl_dataset and walle_skywalker_testset",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action = "Validated" if args.dry_run else "Updated"
    for dataset_root in args.dataset_roots:
        result = add_annotation_rewards(dataset_root, dry_run=args.dry_run)
        print(
            f"{action} {result['dataset']}: {result['frames']} frames in {result['files']} file(s), "
            f"{result['annotated_frames']} annotated frames."
        )
        print(f"  Annotation counts: {result['annotation_counts']}")
        print(f"  Fields: {', '.join(result['columns'])}")


if __name__ == "__main__":
    main()
