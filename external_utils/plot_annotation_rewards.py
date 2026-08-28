#!/usr/bin/env python

"""Plot annotation rewards and discounted returns for the first dataset episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

try:
    from .add_annotation_rewards_to_lerobot_datasets import DEFAULT_DATASET_ROOTS, GAMMAS
except ImportError:
    from add_annotation_rewards_to_lerobot_datasets import DEFAULT_DATASET_ROOTS, GAMMAS


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("annotation_reward_plots")
REWARD_COLUMN = "annotation_reward"
RETURN_COLUMNS = [f"annotation_return_gamma_{gamma}" for gamma in GAMMAS]
COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00")


def load_plot_data(dataset_root: Path):
    data_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No Parquet files found under {dataset_root / 'data'}")

    required = [
        "episode_index",
        "frame_index",
        "fname",
        "annotation",
        REWARD_COLUMN,
        *RETURN_COLUMNS,
    ]
    for path in data_paths:
        missing = set(required) - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return pq.read_table(data_paths, columns=required).to_pandas()


def plot_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    num_episodes: int = 5,
    dpi: int = 180,
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    data = load_plot_data(root)
    episode_indices = sorted(data["episode_index"].unique())[:num_episodes]
    if not episode_indices:
        raise ValueError(f"Dataset contains no episodes: {root}")

    figure, axes = plt.subplots(
        len(episode_indices),
        1,
        figsize=(16, 3.6 * len(episode_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]

    for axis, episode_index in zip(axes, episode_indices, strict=True):
        episode = data[data["episode_index"] == episode_index].sort_values("frame_index")
        frames = episode["frame_index"].to_numpy()

        axis.plot(
            frames,
            episode[REWARD_COLUMN].to_numpy(),
            color="#222222",
            linewidth=1.4,
            drawstyle="steps-mid",
            label="annotation reward",
            zorder=4,
        )
        for column, gamma, color in zip(RETURN_COLUMNS, GAMMAS, COLORS, strict=True):
            axis.plot(
                frames,
                episode[column].to_numpy(),
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=f"return gamma={gamma}",
            )

        positive = episode[episode[REWARD_COLUMN] > 0]
        all_values = episode[[REWARD_COLUMN, *RETURN_COLUMNS]].to_numpy(dtype=np.float32)
        value_min = float(np.nanmin(all_values))
        value_max = float(np.nanmax(all_values))
        value_span = max(value_max - value_min, 1.0)
        axis.set_ylim(value_min - 0.08 * value_span, value_max + 0.28 * value_span)
        label_y = value_max + 0.20 * value_span

        for label_index, row in enumerate(positive.itertuples(index=False)):
            frame = int(row.frame_index)
            reward = float(row.annotation_reward)
            annotation = str(row.annotation)
            axis.axvline(frame, color="#666666", linewidth=0.8, alpha=0.25, zorder=1)
            axis.scatter([frame], [reward], color="#111111", s=28, zorder=5)
            axis.annotate(
                f"{annotation} (+{reward:g})",
                xy=(frame, reward),
                xytext=(frame, label_y - (label_index % 2) * 0.07 * value_span),
                ha="center",
                va="top",
                rotation=90,
                fontsize=8.5,
                color="#111111",
                arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.7},
            )

        fname = str(episode["fname"].iloc[0])
        axis.set_title(f"Episode {int(episode_index)} | {fname}", fontsize=11, loc="left")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Reward / return")
        axis.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.5)
        axis.grid(axis="both", color="#D8D8D8", linewidth=0.6, alpha=0.65)
        axis.set_xlim(int(frames.min()), int(frames.max()))
        axis.legend(loc="upper left", ncol=3, fontsize=8, frameon=True)

    figure.suptitle(
        f"{root.name}: annotation rewards and discounted returns",
        fontsize=15,
        fontweight="bold",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{root.name}_first_{len(episode_indices)}_episodes.png"
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASET_ROOTS),
        help="Dataset roots; defaults to the Plug5 and Walle datasets",
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of initial episodes to plot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_root in args.dataset_roots:
        output_path = plot_dataset(
            dataset_root,
            args.output_dir,
            num_episodes=args.num_episodes,
            dpi=args.dpi,
        )
        print(output_path)


if __name__ == "__main__":
    main()
