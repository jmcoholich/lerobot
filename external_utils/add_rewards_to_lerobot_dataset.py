import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

STEP_REWARD = -1.0
SUCCESS_REWARD = 500.0
PARTIAL_SUCCESS_SCORE = 0.5

# Exactly one of these modes should be enabled.
ALL_EPISODES_SUCCESS = False
ALL_EPISODES_FAILURE = False
ALL_EPISODES_PARTIAL_SUCCESS = False
FNAME_SUCCESS_LABELS = True
FNAME_SUCCESS_LABELS_DEFAULT = 1.0

# Used only when FNAME_SUCCESS_LABELS is True. Unlisted files use
# FNAME_SUCCESS_LABELS_DEFAULT, so only non-success cases need entries here.
fname_is_success = {
    "rickross_plug5_1.h5": 0.5,
    "rickross_plug5_2.h5": 0.0,
    "rickross_plug5_3.h5": 0.0,
    "rickross_plug5_4.h5": 1.0,
    "rickross_plug5_5.h5": 0.0,
    "rickross_plug5_6.h5": 1.0,
    "rickross_plug5_7.h5": 1.0,
    "rickross_plug5_8.h5": 1.0,
    "rickross_plug5_9.h5": 0.5,
    "rickross_plug5_10.h5": 1.0,
    "rickross_plug5_11.h5": 1.0,
    "rickross_plug5_12.h5": 1.0,
    "rickross_plug5_13.h5": 0.0,
    "rickross_plug5_14.h5": 0.0,
    "rickross_plug5_15.h5": 0.5,
    "rickross_plug5_16.h5": 1.0,
    "rickross_plug5_17.h5": 0.0,
    "rickross_plug5_18.h5": 0.0,
    "rickross_plug5_19.h5": 0.0,
    "rickross_plug5_20.h5": 0.5,
    "rickross_plug5_21.h5": 1.0,
    "rickross_plug5_22.h5": 1.0,
    "rickross_plug5_23.h5": 1.0,
    "rickross_plug5_24.h5": 0.0,
    "rickross_plug5_25.h5": 0.5,
    "rickross_plug5_26.h5": 0.0,
    "rickross_plug5_27.h5": 1.0,
    "rickross_plug5_28.h5": 0.5,
    "rickross_plug5_29.h5": 1.0,
    "rickross_plug5_30.h5": 1.0,
    "rickross_plug5_31.h5": 1.0,
    "rickross_plug5_32.h5": 0.0,
    "rickross_plug5_33.h5": 0.0,
    "rickross_plug5_34.h5": 0.5,
    "rickross_plug5_35.h5": 1.0,
    "rickross_plug5_36.h5": 0.0,
    "rickross_plug5_37.h5": 1.0,
    "rickross_plug5_38.h5": 0.0,
    "rickross_plug5_39.h5": 1.0,
    "rickross_plug5_40.h5": 0.5,
    "rickross_plug5_41.h5": 1.0,
    "rickross_plug5_42.h5": 0.0,
    "rickross_plug5_43.h5": 0.5,
    "rickross_plug5_44.h5": 0.5,
    "rickross_plug5_45.h5": 1.0,
    "rickross_plug5_46.h5": 0.5,
    "rickross_plug5_47.h5": 1.0,
    "rickross_plug5_48.h5": 1.0,
    "rickross_plug5_49.h5": 0.5,
    "rickross_plug5_50.h5": 0.0,
    "rickross_plug5_51.h5": 0.0,
    "rickross_plug5_52.h5": 0.0,
    "rickross_plug5_53.h5": 0.5,
    "rickross_plug5_54.h5": 0.0,
    "rickross_plug5_55.h5": 0.5,
    "rickross_plug5_56.h5": 0.5,
    "rickross_plug5_57.h5": 0.0,
    "rickross_plug5_58.h5": 0.0,
    "rickross_plug5_59.h5": 1.0,
    "rickross_plug5_60.h5": 1.0,
    "rickross_plug5_61.h5": 0.0,
    "rickross_plug5_62.h5": 0.0,
    "rickross_plug5_63.h5": 1.0,
    "rickross_plug5_64.h5": 0.5,
    "rickross_plug5_65.h5": 0.5,
    "rickross_plug5_66.h5": 0.5,
    "rickross_plug5_67.h5": 0.5,
    "rickross_plug5_68.h5": 0.5,
    "rickross_plug5_69.h5": 1.0,
    "rickross_plug5_70.h5": 0.0,
    "rickross_plug5_71.h5": 0.5,
    "rickross_plug5_72.h5": 1.0,
    "rickross_plug5_73.h5": 1.0,
    "rickross_plug5_74.h5": 1.0,
    "rickross_plug5_75.h5": 0.5,
    "rickross_plug5_76.h5": 0.5,
    "rickross_plug5_77.h5": 0.0,
    "rickross_plug5_78.h5": 0.0,
    "rickross_plug5_79.h5": 1.0,
    "rickross_plug5_80.h5": 1.0,
    "rickross_plug5_81.h5": 0.0,
    "rickross_plug5_82.h5": 1.0,
    "rickross_plug5_83.h5": 0.5,
    "rickross_plug5_84.h5": 1.0,
    "rickross_plug5_85.h5": 1.0,
    "rickross_plug5_86.h5": 0.0,
    "rickross_plug5_87.h5": 0.0,
    "rickross_plug5_88.h5": 0.5,
    "rickross_plug5_89.h5": 0.5,
    "rickross_plug5_90.h5": 0.5,
    "rickross_plug5_91.h5": 0.0,
    "rickross_plug5_92.h5": 0.5,
    "rickross_plug5_93.h5": 1.0,
    "rickross_plug5_94.h5": 0.0,
    "rickross_plug5_95.h5": 1.0,
    "rickross_plug5_96.h5": 1.0,
    "rickross_plug5_97.h5": 0.0,
    "rickross_plug5_98.h5": 0.5,
    "rickross_plug5_99.h5": 1.0,
    "rickross_plug5_100.h5": 0.0,
}

fname_is_success.update({f"demo_{fname}": success for fname, success in fname_is_success.items()})
GAMMAS = [0.999, 0.995, 0.99, 0.95, 0.9]
STAT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


def reward_return_columns():
    return ["reward", *(f"returns_gamma_{gamma}" for gamma in GAMMAS)]


def get_reward_mode() -> str:
    modes = {
        "ALL_EPISODES_SUCCESS": ALL_EPISODES_SUCCESS,
        "ALL_EPISODES_FAILURE": ALL_EPISODES_FAILURE,
        "ALL_EPISODES_PARTIAL_SUCCESS": ALL_EPISODES_PARTIAL_SUCCESS,
        "FNAME_SUCCESS_LABELS": FNAME_SUCCESS_LABELS,
    }
    active_modes = [mode for mode, enabled in modes.items() if enabled]
    if len(active_modes) != 1:
        raise ValueError(f"Expected exactly one reward mode to be True, got {active_modes}")
    return active_modes[0]


def get_episode_success_scores(episode_last_rows: pd.DataFrame) -> pd.Series:
    reward_mode = get_reward_mode()

    if reward_mode == "ALL_EPISODES_SUCCESS":
        return pd.Series(1.0, index=episode_last_rows.index, dtype=np.float32)

    if reward_mode == "ALL_EPISODES_FAILURE":
        return pd.Series(0.0, index=episode_last_rows.index, dtype=np.float32)

    if reward_mode == "ALL_EPISODES_PARTIAL_SUCCESS":
        return pd.Series(PARTIAL_SUCCESS_SCORE, index=episode_last_rows.index, dtype=np.float32)

    if "fname" not in episode_last_rows.columns:
        raise ValueError("FNAME_SUCCESS_LABELS=True requires an 'fname' column in the parquet data")

    success_scores = episode_last_rows["fname"].map(fname_is_success).fillna(FNAME_SUCCESS_LABELS_DEFAULT)
    return success_scores.astype(np.float32)


def find_dataset_root(parquet_path: Path) -> Path | None:
    for parent in parquet_path.parents:
        if parent.name == "data" and (parent.parent / "meta" / "info.json").exists():
            return parent.parent
    return None


def load_new_feature_values(dataset_root: Path, feature_columns: list[str]) -> dict[str, np.ndarray]:
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet files found under {dataset_root / 'data'}")

    values = {column: [] for column in feature_columns}
    missing_columns_by_file = []

    for data_path in parquet_paths:
        available_columns = set(pq.read_schema(data_path).names)
        missing_columns = [column for column in feature_columns if column not in available_columns]
        if missing_columns:
            missing_columns_by_file.append((data_path.relative_to(dataset_root), missing_columns))
            continue

        columns_df = pd.read_parquet(data_path, columns=feature_columns)
        for column in feature_columns:
            values[column].append(columns_df[column].to_numpy(dtype=np.float32, copy=False))

    if missing_columns_by_file:
        preview = ", ".join(
            f"{path}: {columns}" for path, columns in missing_columns_by_file[:5]
        )
        if len(missing_columns_by_file) > 5:
            preview += f", ... ({len(missing_columns_by_file)} files total)"
        raise ValueError(
            "Not all parquet files have the new reward/return columns yet. "
            f"Metadata was left unchanged. Missing columns: {preview}"
        )

    return {column: np.concatenate(column_values) for column, column_values in values.items()}


def load_json_file(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json_file(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def scalar_feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = values.astype(np.float32, copy=False).reshape(-1)
    stats: dict[str, list[float] | list[int]] = {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values, dtype=np.float64))],
        "std": [float(np.std(values, dtype=np.float64))],
        "count": [int(values.shape[0])],
    }

    for quantile in STAT_QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = [float(np.quantile(values, quantile))]

    return stats


def update_reward_return_metadata(parquet_path: Path, feature_columns: list[str]) -> None:
    dataset_root = find_dataset_root(parquet_path)
    if dataset_root is None:
        print("\nCould not find dataset meta/info.json from parquet path; skipped metadata update.")
        return

    try:
        feature_values = load_new_feature_values(dataset_root, feature_columns)
    except ValueError as exc:
        print(f"\nSkipped metadata update: {exc}")
        return

    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"

    info = load_json_file(info_path)
    for column in feature_columns:
        info["features"][column] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
    write_json_file(info, info_path)

    stats = load_json_file(stats_path) if stats_path.exists() else {}
    for column, values in feature_values.items():
        stats[column] = scalar_feature_stats(values)
    write_json_file(stats, stats_path)

    print(f"\nUpdated metadata at: {dataset_root / 'meta'}")


def inspect_parquet(path, num_rows=5):
    print(f"\n=== Loading: {path} ===\n")

    path = Path(path)
    df = pd.read_parquet(path)
    df["reward"] = STEP_REWARD
    episode_last_rows = df.loc[df.groupby("episode_index")["frame_index"].idxmax()].copy()
    episode_last_rows["success"] = get_episode_success_scores(episode_last_rows)
    df.loc[episode_last_rows.index, "reward"] = SUCCESS_REWARD * episode_last_rows["success"]
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
    feature_cols = reward_return_columns()
    return_cols = feature_cols[1:]
    update_reward_return_metadata(path, feature_cols)

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

    plot_episodes = (
        episode_last_rows.dropna(subset=["success"]).drop_duplicates("success").sort_values("success")
    )
    plot_df = df[df["episode_index"].isin(plot_episodes["episode_index"])]
    success_by_episode = episode_last_rows.set_index("episode_index")["success"].to_dict()
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
        success = success_by_episode[episode_index]
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
    path = "/data3/lerobot_data/plug5_offline_rl_dataset/data/chunk-000/file-000.parquet"
    inspect_parquet(path)
