import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

STEP_REWARD = -1.0
SUCCESS_REWARD = 500.0

fname_is_success = {
"demo_plug3_0.h5": 1.0,
"demo_plug3_10.h5": 1.0,
"demo_plug3_11.h5": 1.0,
"demo_plug3_12.h5": 1.0,
"demo_plug3_13.h5": 1.0,
"demo_plug3_14.h5": 1.0,
"demo_plug3_15.h5": 1.0,
"demo_plug3_16.h5": 1.0,
"demo_plug3_17.h5": 1.0,
"demo_plug3_18.h5": 1.0,
"demo_plug3_19.h5": 1.0,
"demo_plug3_1.h5": 1.0,
"demo_plug3_20.h5": 1.0,
"demo_plug3_21.h5": 1.0,
"demo_plug3_22.h5": 1.0,
"demo_plug3_23.h5": 1.0,
"demo_plug3_24.h5": 1.0,
"demo_plug3_2.h5": 1.0,
"demo_plug3_3.h5": 1.0,
"demo_plug3_4.h5": 1.0,
"demo_plug3_5.h5": 1.0,
"demo_plug3_6.h5": 1.0,
"demo_plug3_7.h5": 1.0,
"demo_plug3_8.h5": 1.0,
"demo_plug3_9.h5": 1.0,
"demo_plug3_rollout_1.h5": 0.0,
"demo_plug3_rollout_2.h5": 0.0,
"demo_plug3_rollout_3.h5": 1.0,
"demo_plug3_rollout_4.h5": 0.0,
"demo_plug3_rollout_5.h5": 0.0,
"demo_plug3_rollout_6.h5": 1.0,
"demo_plug3_rollout_7.h5": 0.0,
"demo_plug3_rollout_8.h5": 1.0,
"demo_plug3_rollout_9.h5": 0.0,
"demo_plug3_rollout_10.h5": 0.0,
"demo_plug3_rollout_11.h5": 0.5,
"demo_plug3_rollout_12.h5": 1.0,
"demo_plug3_rollout_13.h5": 0.0,
"demo_plug3_rollout_14.h5": 1.0,
"demo_plug3_rollout_15.h5": 0.0,
"demo_plug3_rollout_16.h5": 0.5,
"demo_plug3_rollout_17.h5": 1.0,
"demo_plug3_rollout_18.h5": 1.0,
"demo_plug3_rollout_19.h5": 0.0,
"demo_plug3_rollout_20.h5": 0.0,
"demo_plug3_rollout_21.h5": 0.5,
"demo_plug3_rollout_22.h5": 0.0,
"demo_plug3_rollout_23.h5": 1.0,
"demo_plug3_rollout_24.h5": 0.0,
"demo_plug3_rollout_25.h5": 0.0,
}

GAMMAS = [0.999, 0.995, 0.99, 0.95, 0.9]


def _dataset_root_from_path(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return path.parents[2]
    return path


def _stats_json_value(value):
    if np.isnan(value):
        return float("nan")
    return float(value)


def _compute_scalar_column_stats(values):
    values = np.asarray(values, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot compute stats for a column with no finite values")

    quantiles = np.quantile(finite_values, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "min": [_stats_json_value(np.min(finite_values))],
        "max": [_stats_json_value(np.max(finite_values))],
        "mean": [_stats_json_value(np.mean(finite_values))],
        "std": [_stats_json_value(np.std(finite_values))],
        "count": [int(finite_values.size)],
        "q01": [_stats_json_value(quantiles[0])],
        "q10": [_stats_json_value(quantiles[1])],
        "q50": [_stats_json_value(quantiles[2])],
        "q90": [_stats_json_value(quantiles[3])],
        "q99": [_stats_json_value(quantiles[4])],
    }


def update_reward_return_stats(path, feature_names):
    dataset_root = _dataset_root_from_path(path)
    stats_path = dataset_root / "meta" / "stats.json"
    data_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    with stats_path.open("r") as f:
        stats = json.load(f)

    values_by_feature = {name: [] for name in feature_names}
    for data_path in data_paths:
        df = pd.read_parquet(data_path, columns=feature_names)
        for name in feature_names:
            values_by_feature[name].append(df[name].to_numpy(dtype=np.float32, copy=False))

    for name, chunks in values_by_feature.items():
        stats[name] = _compute_scalar_column_stats(np.concatenate(chunks))

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=4, allow_nan=True)
        f.write("\n")


def update_info_features(path, feature_names):
    info_path = _dataset_root_from_path(path) / "meta" / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    for name in feature_names:
        info["features"][name] = {"dtype": "float32", "shape": [1], "names": None}
    with info_path.open("w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def inspect_parquet(path, num_rows=5):
    print(f"\n=== Loading: {path} ===\n")

    df = pd.read_parquet(path)
    df["reward"] = STEP_REWARD
    last_frame = df.groupby("episode_index")["frame_index"].transform("max")
    is_last_frame = df["frame_index"] == last_frame
    df.loc[is_last_frame, "reward"] = SUCCESS_REWARD * df.loc[is_last_frame, "fname"].map(fname_is_success)
    df["reward"] = df["reward"].astype("float32")

    rewards = df["reward"].to_numpy(dtype=np.float32, copy=False)
    episode_indices = df["episode_index"].to_numpy(copy=False)
    gammas = np.asarray(GAMMAS, dtype=np.float32)
    returns = np.empty((len(df), len(gammas)), dtype=np.float32)
    running_returns = np.zeros(len(gammas), dtype=np.float32)

    for i in range(len(df) - 1, -1, -1):
        if i == len(df) - 1 or episode_indices[i] != episode_indices[i + 1]:
            running_returns.fill(0.0)
        running_returns = rewards[i] + gammas * running_returns
        returns[i] = running_returns

    for gamma_idx, gamma in enumerate(GAMMAS):
        df[f"returns_gamma_{gamma}"] = returns[:, gamma_idx]

    df.to_parquet(path, index=False)
    reward_return_features = ["reward", *[f"returns_gamma_{gamma}" for gamma in GAMMAS]]
    update_info_features(path, reward_return_features)
    update_reward_return_stats(path, reward_return_features)

    print("=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(list(df.columns))

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== SAMPLE ROWS ===")
    print(df.head(num_rows))

    print("\n=== FIRST ROW (FULL) ===")
    first = df.iloc[0]
    for col in df.columns:
        val = first[col]
        print(f"\n--- {col} ---")
        print(type(val))
        try:
            print(val if len(str(val)) < 500 else str(val)[:500] + "...")
        except:
            print(val)

    episode_last_rows = df.loc[df.groupby("episode_index")["frame_index"].idxmax()].copy()
    episode_last_rows["success"] = episode_last_rows["fname"].map(fname_is_success)
    plot_episodes = (
        episode_last_rows[episode_last_rows["success"].isin([0.0, 0.5, 1.0])]
        .drop_duplicates("success")
        .sort_values("success")
    )
    plot_df = df[df["episode_index"].isin(plot_episodes["episode_index"])]
    return_cols = [f"returns_gamma_{gamma}" for gamma in GAMMAS]
    fig, axes = plt.subplots(len(plot_episodes), 1, sharex=False, figsize=(12, 4 * len(plot_episodes)))
    axes = np.atleast_1d(axes)
    max_episode_steps = plot_df.groupby("episode_index").size().max() + 50
    y_values = plot_df[["reward", *return_cols]].to_numpy(copy=False)
    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    y_padding = max((y_max - y_min) * 0.05, 1.0)
    for ax, (episode_index, episode_df) in zip(axes, plot_df.groupby("episode_index", sort=False)):
        x = np.arange(len(episode_df))
        ax.plot(x, episode_df["reward"].to_numpy(copy=False), label="reward", color="black", linewidth=2)
        for col in return_cols:
            ax.plot(x, episode_df[col].to_numpy(copy=False), label=col, linestyle=":", linewidth=2.5, alpha=0.85)
        success = fname_is_success[episode_df["fname"].iloc[-1]]
        ax.set_title(f"Episode {episode_index} (success {success})")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.set_xlim(0, max_episode_steps - 1)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = Path(__file__).with_name(f"{Path(path).stem}_rewards_returns_by_success.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved reward/return plot to: {plot_path}")
    # breakpoint()


if __name__ == "__main__":
    # data_dir = Path("/data3/lerobot_data/plug3_w_rollouts/data")  # ripl-d3
    data_dir = Path("/coc/testnvme/jcoholich3/lerobot_data/plug3_w_rollouts/data")  # skynet
    chunk_dirs = [entry for entry in data_dir.iterdir() if entry.is_dir()]
    if len(chunk_dirs) != 1:
        raise ValueError(f"Expected exactly one chunk dir under {data_dir}, found {len(chunk_dirs)}")
    path = chunk_dirs[0] / "file-000.parquet"
    inspect_parquet(path)
