#!/usr/bin/env python
"""Keep whole episodes using strict success annotations or StackFour rewards.

Videos are hard-linked without decoding or reencoding; source and destination
must be on the same filesystem. Treat linked video files as immutable. Shared
video chunks can still contain unused footage, which the subset never indexes.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats

try:
    from .add_annotations_to_lerobot_dataset import find_data_files, load_json, write_json_atomic
except ImportError:
    from add_annotations_to_lerobot_dataset import find_data_files, load_json, write_json_atomic

DEFAULT_SOURCE = Path(
    "/coc/testnvme/jcoholich3/lerobot_data/plug5_offline_rl_dataset_walle_skywalker_testset_annotated"
)


def feature_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    stats = {name: getattr(values, name)(axis=0).tolist() for name in ("min", "max", "mean", "std")}
    stats["count"] = [len(values)]
    for q in (1, 10, 50, 90, 99):
        stats[f"q{q:02d}"] = np.percentile(values, q, axis=0).tolist()
    return stats


def select_success_episodes(data, stack_four_reward_key=None):
    if stack_four_reward_key is not None:
        episode_ids = np.asarray(data["episode_index"])
        rewards = np.asarray(data[stack_four_reward_key].to_pylist(), dtype=np.float64)
        if rewards.size != len(data):
            raise ValueError("Expected one scalar reward per frame")
        rewards = rewards.reshape(-1)
        selected = []
        for episode in np.unique(episode_ids):
            values = rewards[episode_ids == episode]
            picks = np.isclose(values, 0.135, atol=1e-6, rtol=0)
            places = (values >= 0.27 - 1e-6) & (values <= 0.405 + 1e-6)
            if (
                np.isfinite(values).all()
                and (values >= 0).all()
                and picks.sum() == 3
                and places.sum() == 3
                and np.all((values == 0) | picks | places)
            ):
                selected.append(int(episode))
        return selected
    outcomes = {}
    for episode, annotation in zip(
        data["episode_index"].to_pylist(), data["annotation"].to_pylist(), strict=False
    ):
        label = (annotation or "").strip()
        if label and label != "plug picked":
            outcomes.setdefault(episode, set()).add(label)
    return sorted(episode for episode, labels in outcomes.items() if labels == {"success"})


def create_subset(source: Path, output: Path, dry_run=False, stack_four_reward_key=None):
    info = load_json(source / "meta/info.json")
    if info["codebase_version"] != "v3.0":
        raise ValueError("This script requires a LeRobot v3.0 dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    data = pa.concat_tables([pq.read_table(path) for path in find_data_files(source)])
    selected = select_success_episodes(data, stack_four_reward_key)
    if not selected:
        raise ValueError("No strict-success episodes found")
    data = data.filter(pc.is_in(data["episode_index"], value_set=pa.array(selected)))
    data = data.sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
    print(f"Keeping {len(selected):,}/{info['total_episodes']:,} episodes, {len(data):,} frames", flush=True)
    if dry_run:
        return

    episodes = pa.concat_tables(
        [pq.read_table(path) for path in sorted((source / "meta/episodes").rglob("*.parquet"))]
    ).to_pylist()
    episodes = {row["episode_index"]: row for row in episodes}
    mapping = {old: new for new, old in enumerate(selected)}
    for key, values in (
        ("episode_index", [mapping[x] for x in data["episode_index"].to_pylist()]),
        ("index", np.arange(len(data))),
    ):
        field = data.schema.field(key)
        data = data.set_column(data.schema.get_field_index(key), field, pa.array(values, type=field.type))
    # Recompute numeric stats, including annotation returns and AWR columns added
    # after recording. Video stats can be aggregated without decoding images.
    numeric = {
        key: np.asarray(data[key].to_pylist())
        for key, feature in info["features"].items()
        if key in data.column_names and feature["dtype"] not in ("string", "video", "image")
    }
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        rows, video_stats, provenance = [], [], []
        start = 0
        for old, new in mapping.items():
            row = episodes[old].copy()
            end = start + row["length"]
            if not np.all(numeric["episode_index"][start:end] == new) or end > len(data):
                raise ValueError(f"Frame count disagrees with metadata for episode {old}")
            row.update(
                {
                    "episode_index": new,
                    "dataset_from_index": start,
                    "dataset_to_index": end,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            for key, values in numeric.items():
                for stat, value in feature_stats(values[start:end]).items():
                    row[f"stats/{key}/{stat}"] = value
            stats = {}
            for key in video_keys:
                relative = info["video_path"].format(
                    video_key=key,
                    chunk_index=row[f"videos/{key}/chunk_index"],
                    file_index=row[f"videos/{key}/file_index"],
                )
                target = root / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source / relative, target)
                prefix = f"stats/{key}/"
                stats[key] = {k[len(prefix) :]: np.asarray(v) for k, v in row.items() if k.startswith(prefix)}
            video_stats.append(stats)
            rows.append(row)
            provenance.append({"episode_index": new, "source_episode_index": old})
            start = end
        if start != len(data):
            raise ValueError("Frame count disagrees with episode metadata")
        stats = {
            k: {s: v.tolist() for s, v in values.items()}
            for k, values in aggregate_stats(video_stats).items()
        }
        stats.update({key: feature_stats(values) for key, values in numeric.items()})
        pq.write_table(data.replace_schema_metadata(None), root / "data/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
        shutil.copy2(source / "meta/tasks.parquet", root / "meta/tasks.parquet")
        info.update(
            total_episodes=len(selected), total_frames=len(data), splits={"train": f"0:{len(selected)}"}
        )
        write_json_atomic(info, root / "meta/info.json")
        write_json_atomic(stats, root / "meta/stats.json")
        selection = {"method": "strict_success_annotation"}
        if stack_four_reward_key is not None:
            selection = {
                "method": "stack_four_rewards",
                "reward_key": stack_four_reward_key,
                "pick_reward": 0.135,
                "place_reward_range": [0.27, 0.405],
                "absolute_tolerance": 1e-6,
                "required_picks": 3,
                "required_places": 3,
                "allow_negative_or_other_rewards": False,
            }
        write_json_atomic(
            {"source": str(source), "selection": selection, "episodes": provenance},
            root / "meta/success_subset.json",
        )
        root.rename(output)
    print(f"Created {output} (videos hard-linked; no reencoding)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stack-four-reward-key",
        help="Select exactly three 0.135 picks and three 0.27–0.405 places, with no negative or other rewards",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = (args.output or source.with_name(source.name + "_SUCCESS_ONLY")).expanduser().resolve()
    create_subset(source, output, args.dry_run, args.stack_four_reward_key)


if __name__ == "__main__":
    main()
