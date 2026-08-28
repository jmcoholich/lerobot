import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GAMMAS = [0.999, 0.995, 0.99, 0.95, 0.9, 1.0]
TERMINAL_REWARD_TO_OUTCOME = {
    0.0: "failure",
    250.0: "partial success",
    500.0: "success",
}


def plot_reward_series(
    df: pd.DataFrame,
    reward_column: str,
    return_columns: list[str],
    output_path: Path,
) -> None:
    terminal_indices = df.groupby("episode_index")["frame_index"].idxmax()
    terminal_rows = df.loc[terminal_indices].copy()
    terminal_rows["outcome"] = terminal_rows["reward"].map(
        TERMINAL_REWARD_TO_OUTCOME
    )
    invalid_rows = terminal_rows[terminal_rows["outcome"].isna()]
    if not invalid_rows.empty:
        details = ", ".join(
            f"episode {row.episode_index}: {row.reward}"
            for row in invalid_rows.itertuples()
        )
        raise ValueError(f"Unexpected terminal reward(s): {details}")

    plot_episodes = (
        terminal_rows.sort_values("episode_index")
        .head(6)
        .set_index("episode_index")["outcome"]
        .to_dict()
    )
    plot_df = df[df["episode_index"].isin(plot_episodes)].sort_values(
        ["episode_index", "frame_index"]
    )
    figure, axes = plt.subplots(
        len(plot_episodes),
        1,
        sharex=False,
        figsize=(12, 4 * len(plot_episodes)),
    )
    axes = np.atleast_1d(axes)
    max_episode_steps = plot_df.groupby("episode_index").size().max() + 50
    y_values = plot_df[[reward_column, *return_columns]].to_numpy(copy=False)
    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    y_padding = max((y_max - y_min) * 0.05, 0.01)

    for axis, (episode_index, episode_df) in zip(
        axes, plot_df.groupby("episode_index", sort=False)
    ):
        x = np.arange(len(episode_df))
        axis.plot(
            x,
            episode_df[reward_column].to_numpy(copy=False),
            label=reward_column,
            color="black",
            linewidth=2,
        )
        for column in return_columns:
            axis.plot(
                x,
                episode_df[column].to_numpy(copy=False),
                label=column,
                linestyle=":",
                linewidth=2.5,
                alpha=0.85,
            )
        axis.set_title(f"Episode {episode_index} ({plot_episodes[episode_index]})")
        axis.set_xlabel("step")
        axis.set_ylabel("value")
        axis.set_xlim(0, max_episode_steps - 1)
        axis.set_ylim(y_min - y_padding, y_max + y_padding)
        axis.legend(loc="best")
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {output_path}")


def plot_dataset(dataset_root: Path, output_dir: Path) -> None:
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet files found under {dataset_root / 'data'}")

    schemas = [set(pq.read_schema(path).names) for path in parquet_paths]
    required_columns = {
        "episode_index",
        "frame_index",
        "reward",
        "sparse_reward",
    }
    dense_returns = [
        f"returns_gamma_{gamma}"
        for gamma in GAMMAS
        if all(f"returns_gamma_{gamma}" in schema for schema in schemas)
    ]
    sparse_returns = [
        f"sparse_returns_gamma_{gamma}"
        for gamma in GAMMAS
        if all(f"sparse_returns_gamma_{gamma}" in schema for schema in schemas)
    ]
    requested_returns = {
        *(f"returns_gamma_{gamma}" for gamma in GAMMAS),
        *(f"sparse_returns_gamma_{gamma}" for gamma in GAMMAS),
    }
    skipped_returns = sorted(
        column
        for column in requested_returns
        if not all(column in schema for schema in schemas)
    )
    if skipped_returns:
        print(f"Skipping missing return columns in {dataset_root}: {skipped_returns}")

    columns = [
        "episode_index",
        "frame_index",
        "reward",
        *dense_returns,
        "sparse_reward",
        *sparse_returns,
    ]
    dataframes = []
    for path, schema in zip(parquet_paths, schemas):
        missing = sorted(required_columns.difference(schema))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        dataframes.append(pd.read_parquet(path, columns=columns))
    df = pd.concat(dataframes, ignore_index=True)

    plot_reward_series(
        df,
        "reward",
        dense_returns,
        output_dir
        / f"{dataset_root.name}_rewards_returns_by_success.png",
    )
    plot_reward_series(
        df,
        "sparse_reward",
        sparse_returns,
        output_dir
        / f"{dataset_root.name}_sparse_rewards_returns_by_success.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot existing dense and sparse rewards without modifying datasets."
    )
    parser.add_argument(
        "dataset_roots",
        nargs="+",
        type=Path,
        help="Dataset root directories containing data/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for generated PNGs (default: this script's directory)",
    )
    args = parser.parse_args()

    for dataset_root in args.dataset_roots:
        plot_dataset(dataset_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
