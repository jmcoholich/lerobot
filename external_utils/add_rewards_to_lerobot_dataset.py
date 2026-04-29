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


def update_info_features(path, feature_names):
    info_path = Path(path).parents[2] / "meta" / "info.json"
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
    update_info_features(path, ["reward", *[f"returns_gamma_{gamma}" for gamma in GAMMAS]])

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
    path = "/data3/lerobot_data/plug3_w_rollouts/data/chunk-000/file-000.parquet"
    inspect_parquet(path)
