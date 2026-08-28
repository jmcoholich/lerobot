import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


GAMMAS = [0.999, 0.995, 0.99, 0.95, 0.9, 1.0]
TERMINAL_REWARD_TO_SPARSE_REWARD = {
    500.0: 1.0,
    250.0: 0.5,
    0.0: 0.0,
}


def sparse_feature_columns() -> list[str]:
    return [
        "sparse_reward",
        *(f"sparse_returns_gamma_{gamma}" for gamma in GAMMAS),
    ]


def add_sparse_rewards(dataset_root: Path) -> None:
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet files found under {dataset_root / 'data'}")

    dataframes = [pd.read_parquet(path) for path in parquet_paths]
    new_columns = sparse_feature_columns()
    for path, dataframe in zip(parquet_paths, dataframes):
        missing = {"episode_index", "frame_index", "reward"}.difference(dataframe.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    inconsistent_columns = [
        column
        for column in new_columns
        if len({column in dataframe.columns for dataframe in dataframes}) > 1
    ]
    if inconsistent_columns:
        raise ValueError(
            "Sparse columns are inconsistent across parquet shards: "
            f"{inconsistent_columns}"
        )
    columns_to_add = [
        column for column in new_columns if column not in dataframes[0].columns
    ]
    if not columns_to_add:
        print(f"{dataset_root} already contains all requested sparse columns")
        return

    lengths = [len(dataframe) for dataframe in dataframes]
    df = pd.concat(dataframes, ignore_index=True)
    terminal_indices = df.groupby("episode_index")["frame_index"].idxmax()
    terminal_rewards = df.loc[terminal_indices, "reward"]
    invalid_rewards = terminal_rewards[
        ~terminal_rewards.isin(TERMINAL_REWARD_TO_SPARSE_REWARD)
    ]
    if not invalid_rewards.empty:
        details = ", ".join(
            f"episode {df.at[index, 'episode_index']}: {reward}"
            for index, reward in invalid_rewards.items()
        )
        raise ValueError(f"Unexpected terminal reward(s): {details}")

    if "sparse_reward" in columns_to_add:
        sparse_rewards = np.zeros(len(df), dtype=np.float32)
        sparse_rewards[terminal_indices.to_numpy()] = terminal_rewards.map(
            TERMINAL_REWARD_TO_SPARSE_REWARD
        ).to_numpy(dtype=np.float32)
        df["sparse_reward"] = sparse_rewards
    else:
        sparse_rewards = df["sparse_reward"].to_numpy(dtype=np.float32, copy=False)

    missing_gammas = [
        gamma
        for gamma in GAMMAS
        if f"sparse_returns_gamma_{gamma}" in columns_to_add
    ]
    gammas = np.asarray(missing_gammas, dtype=np.float32)
    returns = np.empty((len(df), len(gammas)), dtype=np.float32)
    for _, episode_df in df.groupby("episode_index", sort=False):
        ordered_indices = episode_df.sort_values("frame_index").index.to_numpy()
        running_returns = np.zeros(len(gammas), dtype=np.float32)
        for index in ordered_indices[::-1]:
            running_returns = sparse_rewards[index] + gammas * running_returns
            returns[index] = running_returns

    for gamma_index, gamma in enumerate(missing_gammas):
        df[f"sparse_returns_gamma_{gamma}"] = returns[:, gamma_index]

    temporary_paths = []
    start = 0
    try:
        for path, length in zip(parquet_paths, lengths):
            shard = df.iloc[start : start + length]
            start += length
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(fd)
            temporary_path = Path(temporary_name)
            temporary_paths.append(temporary_path)
            shard.to_parquet(temporary_path, index=False)

        for temporary_path, path in zip(temporary_paths, parquet_paths):
            os.replace(temporary_path, path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    update_metadata(dataset_root, df, columns_to_add)
    counts = terminal_rewards.value_counts().to_dict()
    print(
        f"Added {columns_to_add} to {dataset_root}: {len(terminal_indices)} episodes "
        f"({counts.get(500.0, 0)} success, "
        f"{counts.get(250.0, 0)} partial, {counts.get(0.0, 0)} failure)"
    )


def feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = values.astype(np.float32, copy=False)
    stats: dict[str, list[float] | list[int]] = {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values, dtype=np.float64))],
        "std": [float(np.std(values, dtype=np.float64))],
        "count": [int(values.shape[0])],
    }
    for quantile in [0.01, 0.10, 0.50, 0.90, 0.99]:
        stats[f"q{int(quantile * 100):02d}"] = [
            float(np.quantile(values, quantile))
        ]
    return stats


def update_metadata(
    dataset_root: Path, df: pd.DataFrame, feature_columns: list[str]
) -> None:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"

    with info_path.open() as file:
        info = json.load(file)
    for column in feature_columns:
        info["features"][column] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
    with info_path.open("w") as file:
        json.dump(info, file, indent=4, ensure_ascii=False)

    if stats_path.exists():
        with stats_path.open() as file:
            stats = json.load(file)
    else:
        stats = {}
    for column in feature_columns:
        stats[column] = feature_stats(
            df[column].to_numpy(dtype=np.float32, copy=False)
        )
    with stats_path.open("w") as file:
        json.dump(stats, file, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add sparse rewards and discounted returns to LeRobot datasets."
    )
    parser.add_argument(
        "dataset_roots",
        nargs="+",
        type=Path,
        help="Dataset root directories containing data/ and meta/",
    )
    args = parser.parse_args()

    for dataset_root in args.dataset_roots:
        add_sparse_rewards(dataset_root.resolve())


if __name__ == "__main__":
    main()
