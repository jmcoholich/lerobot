#!/usr/bin/env python
"""Validate OOF inference inputs and merge raw values/advantages into a LeRobot dataset."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def scalar_column(table, key, dtype=np.float64):
    values = np.asarray(table[key].to_pylist(), dtype=dtype)
    if values.size != len(table) or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite scalar per row for {key}")
    return values.reshape(-1)


def dataset_rows(root, value_key):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquet files in {root}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["index", "episode_index", "frame_index", value_key]) for path in files]
    )
    rows = {key: scalar_column(table, key, np.int64) for key in ("index", "episode_index", "frame_index")}
    rows[value_key] = scalar_column(table, value_key)
    if not np.array_equal(np.sort(rows["index"]), np.arange(len(table))):
        raise ValueError("Dataset global indices must be unique and cover every frame")
    for episode in np.unique(rows["episode_index"]):
        frames = rows["frame_index"][rows["episode_index"] == episode]
        if not np.array_equal(np.sort(frames), np.arange(len(frames))):
            raise ValueError(f"Missing or duplicated frame indices in episode {episode}")
    return files, rows


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        **{f"q{q:02d}": float(np.percentile(values, q)) for q in (1, 10, 25, 50, 75, 90, 99)},
        "positive_fraction": float((values > 0).mean()),
        "negative_fraction": float((values < 0).mean()),
    }


def feature_stats(values):
    stats = summary(values)
    return {
        key: [stats[key]] for key in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
    }


def prepare(args):
    root = args.dataset_root.resolve()
    info = read_json(root / "meta/info.json")
    episodes = set(range(info["total_episodes"]))
    folds, held_out, target_settings = [], set(), None
    for fold in range(1, 6):
        run = args.run_template.format(fold=fold)
        path = (REPO / "outputs" / run / "checkpoints" / args.checkpoint / "pretrained_model").resolve()
        cfg, train = read_json(path / "config.json"), read_json(path / "train_config.json")
        if not (path / "model.safetensors").is_file():
            raise ValueError(f"Missing weights in {path}")
        if not cfg.get("use_value_model") or cfg.get("value_dim", 1) != 1:
            raise ValueError(f"Not a scalar value model: {path}")
        if cfg.get("value_bootstrap_steps", 0):
            raise ValueError(f"{path} used bootstrapped targets; stored returns are not its training targets")
        settings = {key: cfg[key] for key in ("value_key", "value_discount", "value_reward_key")}
        if target_settings is not None and settings != target_settings:
            raise ValueError("Fold checkpoints use different return targets")
        target_settings = settings
        if any(train["policy"].get(key) != val for key, val in settings.items()):
            raise ValueError(f"Saved policy and training target settings disagree: {path}")
        if Path(train["dataset"]["root"]).resolve() != root:
            raise ValueError(f"{path} was trained on a different dataset: {train['dataset']['root']}")
        train_eps, test_eps = train["train_episodes"], train["test_episodes"]
        if (
            not train_eps
            or not test_eps
            or len(set(train_eps)) != len(train_eps)
            or len(set(test_eps)) != len(test_eps)
        ):
            raise ValueError(f"Empty or duplicated fold membership: {path}")
        if set(train_eps) & set(test_eps) or set(train_eps) | set(test_eps) != episodes:
            raise ValueError(f"Train/test split is not a disjoint partition of the dataset: {path}")
        if held_out & set(test_eps):
            raise ValueError("An episode is held out by more than one fold")
        held_out.update(test_eps)
        weight_stat = (path / "model.safetensors").stat()
        folds.append(
            {
                "run": run,
                "policy_path": str(path),
                "episodes": sorted(test_eps),
                "prediction_key": f"{path.parents[2].name}_{path.parent.name}",
                "weights_signature": [weight_stat.st_size, weight_stat.st_mtime_ns],
                "manifest": f"fold_{fold}",
            }
        )
    if held_out != episodes:
        raise ValueError("Held-out folds do not cover the entire dataset")
    files, rows = dataset_rows(root, target_settings["value_key"])
    if len(rows["index"]) != info["total_frames"] or set(rows["episode_index"]) != episodes:
        raise ValueError("Dataset frame/episode coverage disagrees with meta/info.json")
    name = (
        args.name
        or f"{args.run_template.replace('_fold_{fold}', '').replace('{fold}', 'folds')}_{args.checkpoint}"
    )
    if not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name
    ):
        raise ValueError("Use --name containing only letters, digits, underscores or hyphens")
    columns = {kind: f"awr_{kind}_{name}_OOF" for kind in ("value", "advantage")}
    for path in files:
        collisions = set(columns.values()) & set(pq.read_schema(path).names)
        if collisions:
            raise ValueError(f"Columns already exist: {sorted(collisions)}; choose a new --name")
    counts = np.unique(rows["episode_index"], return_counts=True)[1]
    return {
        "dataset_root": str(root),
        "name": name,
        "columns": columns,
        "folds": folds,
        **target_settings,
        "dataset": {
            "episodes": len(episodes),
            "frames": len(rows["index"]),
            "fps": info["fps"],
            "duration_hours": len(rows["index"]) / info["fps"] / 3600,
            "episode_lengths_frames": summary(counts),
            "returns": summary(rows[target_settings["value_key"]]),
        },
    }


def write_plan(args):
    plan = prepare(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("Validated; no dataset changes.")
        return
    # Plain tab-separated fields let Bash read the validated inputs without evaluating shell code.
    fold_rows = [
        (fold["run"], fold["policy_path"], fold["manifest"], ",".join(map(str, fold["episodes"])))
        for fold in plan["folds"]
    ]
    if any("\t" in field or "\n" in field for row in fold_rows for field in row):
        raise ValueError("Run names and checkpoint paths cannot contain tabs or newlines")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{plan['name']}_", dir=args.output_dir.resolve()))
    plan["output_dir"] = str(output)
    write_json(output / "plan.json", plan)
    (output / "folds.tsv").write_text("".join("\t".join(row) + "\n" for row in fold_rows))
    print(output)


def collect(plan, rows):
    root, output = Path(plan["dataset_root"]), Path(plan["output_dir"])
    values = np.full(len(rows["index"]), np.nan)
    for fold in plan["folds"]:
        print(f"Validating {fold['manifest']}: {len(fold['episodes'])} episode predictions", flush=True)
        path = Path(fold["policy_path"])
        weight_stat = (path / "model.safetensors").stat()
        if [weight_stat.st_size, weight_stat.st_mtime_ns] != fold["weights_signature"]:
            raise ValueError(f"Checkpoint weights changed after submission: {path}")
        for episode in fold["episodes"]:
            matches = list((output / fold["manifest"]).glob(f"task_*/episode_{episode}/all_cams.json"))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one prediction file for {fold['manifest']} episode {episode}"
                )
            data = read_json(matches[0])
            prediction, target = data[fold["prediction_key"]], data[plan["value_key"]]
            if Path(prediction["model"]).resolve() != path or prediction["value_key"] != plan["value_key"]:
                raise ValueError(f"Wrong model/return key in {matches[0]}")
            if prediction["prediction_scale"] not in ("raw", "model_output"):
                raise ValueError(f"Unexpected value scale in {matches[0]}")
            if (
                prediction["prediction_scale"] == "model_output"
                and prediction["unnormalized_from"] is not None
            ):
                raise ValueError(f"Values are still normalized in {matches[0]}")
            if Path(target["source"]).resolve() != root:
                raise ValueError(f"Wrong return source in {matches[0]}")
            positions = np.flatnonzero(rows["episode_index"] == episode)
            positions = positions[np.argsort(rows["frame_index"][positions])]
            expected = rows["frame_index"][positions].tolist()
            if prediction["frame_indices"] != expected or target["frame_indices"] != expected:
                raise ValueError(f"Missing, duplicate, sampled or misordered frames in {matches[0]}")
            pred = np.asarray(prediction["values"], dtype=np.float64)
            targets = np.asarray(target["values"], dtype=np.float64)
            if pred.shape != (len(positions),) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid predictions in {matches[0]}")
            if targets.shape != pred.shape or not np.allclose(
                targets, rows[plan["value_key"]][positions], rtol=1e-6, atol=1e-7
            ):
                raise ValueError(f"Inference returns differ from the dataset in {matches[0]}")
            if np.isfinite(values[positions]).any():
                raise ValueError(f"Duplicate OOF predictions for episode {episode}")
            values[positions] = pred
    if not np.isfinite(values).all():
        raise ValueError("OOF predictions do not cover all dataset frames")
    return {
        plan["columns"]["value"]: values.astype(np.float32),
        plan["columns"]["advantage"]: (rows[plan["value_key"]] - values).astype(np.float32),
    }


def plot_histograms(plan, series):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    fig = plt.figure(figsize=(16, 5), layout="constrained")
    grid = fig.add_gridspec(1, 3)
    for slot, vals, label in zip(
        grid, series.values(), ("Training returns", "OOF raw values", "OOF advantages"), strict=True
    ):
        counts, edges = np.histogram(vals, bins=80)
        peaks = counts[counts > 40_000]
        upper_start = float(peaks.min() - counts.max() * 0.08) if len(peaks) else 0
        if upper_start > 40_000:
            split = slot.subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
            top = fig.add_subplot(split[0])
            bottom = fig.add_subplot(split[1], sharex=top)
            top.set_ylim(upper_start, counts.max() * 1.08)
            bottom.set_ylim(0, 40_000)
            bottom.set_yticks([0, 10_000, 20_000, 30_000, 40_000])
            top.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            top.spines["bottom"].set_visible(False)
            bottom.spines["top"].set_visible(False)
            top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            for ax, y in ((top, 0), (bottom, 1)):
                ax.plot(
                    [0, 1],
                    [y, y],
                    transform=ax.transAxes,
                    linestyle="none",
                    color="black",
                    marker=[(-1, -0.5), (1, 0.5)],
                    markersize=10,
                    clip_on=False,
                )
            axes = (top, bottom)
        else:
            top = bottom = fig.add_subplot(slot)
            bottom.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            axes = (bottom,)
        for ax in axes:
            ax.stairs(counts, edges, fill=True)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        top.set_title(label)
        top.text(
            0.97,
            0.95,
            f"Min: {np.min(vals):.6g}\nMax: {np.max(vals):.6g}",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        bottom.set(xlabel="Raw return units", ylabel="Frames")
    fig.suptitle(plan["name"])
    fig.savefig(Path(plan["output_dir"]) / "histograms.png", dpi=160)
    plt.close(fig)


def report(plan, rows, columns):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {plan["value_key"]: rows[plan["value_key"]], **columns}
    stats = {
        "dataset": plan["dataset"],
        "value_key": plan["value_key"],
        "columns": {key: summary(val) for key, val in series.items()},
    }
    advantage = columns[plan["columns"]["advantage"]].astype(np.float64)
    stats["value_error"] = {
        "mae": float(np.abs(advantage).mean()),
        "rmse": float(np.sqrt((advantage**2).mean())),
    }
    stats["folds"] = [
        {
            "run": fold["run"],
            "episodes": len(fold["episodes"]),
            "columns": {
                key: summary(val[np.isin(rows["episode_index"], fold["episodes"])])
                for key, val in series.items()
            },
        }
        for fold in plan["folds"]
    ]
    output = Path(plan["output_dir"])
    write_json(output / "stats.json", stats)
    plot_histograms(plan, series)

    episodes = np.unique(rows["episode_index"])[:5]
    dataset = pa_ds.dataset(Path(plan["dataset_root"]) / "data", format="parquet")
    filenames = {int(episode): "Not recorded" for episode in episodes}
    if "fname" in dataset.schema.names:
        metadata = dataset.to_table(
            columns=["episode_index", "fname"], filter=pa_ds.field("episode_index").isin(episodes.tolist())
        ).to_pydict()
        for episode in episodes:
            names = {
                name
                for ep, name in zip(metadata["episode_index"], metadata["fname"], strict=True)
                if ep == episode and name
            }
            filenames[int(episode)] = ", ".join(sorted(names)) or "Not recorded"
    plots = output / "episode_plots"
    plots.mkdir(exist_ok=True)
    for episode in episodes:
        print(f"Plotting episode {episode}", flush=True)
        positions = np.flatnonzero(rows["episode_index"] == episode)
        positions = positions[np.argsort(rows["frame_index"][positions])]
        fold = next(fold for fold in plan["folds"] if episode in fold["episodes"])
        fig, (ax, details) = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]}, layout="constrained"
        )
        for vals, label, style in zip(
            series.values(),
            ("Ground-truth return", "OOF predicted value", "Advantage (return - value)"),
            ("-", "-", "--"),
            strict=True,
        ):
            ax.plot(rows["frame_index"][positions], vals[positions], style, label=label, linewidth=1.3)
        ax.set(title=f"Episode {episode}", xlabel="Frame index", ylabel="Raw return units")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.grid(alpha=0.2)
        ax.legend()
        episode_advantages = advantage[positions]
        notes = [
            f"Dataset: {Path(plan['dataset_root']).name}",
            f"Source filename: {filenames[int(episode)]}",
            f"Dataset episode index: {episode} (zero-based)",
            f"OOF model/checkpoint: {fold['prediction_key']}",
            f"Return column: {plan['value_key']}",
            f"Frames: {len(positions)} | FPS: {plan['dataset']['fps']} | Duration: {len(positions) / plan['dataset']['fps']:.2f} s",
            f"Value MAE: {np.abs(episode_advantages).mean():.4g} | RMSE: {np.sqrt((episode_advantages**2).mean()):.4g}",
        ]
        details.axis("off")
        details.text(
            0,
            1,
            "\n\n".join(textwrap.fill(note, 52) for note in notes),
            va="top",
            transform=details.transAxes,
            fontsize=10,
        )
        fig.savefig(plots / f"episode_{episode:06d}.png", dpi=160)
        plt.close(fig)
    return stats


def add_columns(table, columns):
    for key, vals in columns.items():
        table = table.append_column(key, pa.array(vals, type=pa.float32()))
    # Keep Hugging Face feature metadata consistent with the physical parquet schema.
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        for key in columns:
            hf["info"]["features"][key] = {"dtype": "float32", "_type": "Value"}
        metadata[b"huggingface"] = json.dumps(hf).encode()
        table = table.replace_schema_metadata(metadata)
    return table


def merge(plan_path):
    plan = read_json(plan_path)
    root = Path(plan["dataset_root"])
    # One writer for this dataset; Slurm dependencies keep inference readers out of this phase.
    with (root / "meta/.awr.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Reading dataset: {root}", flush=True)
        files, rows = dataset_rows(root, plan["value_key"])
        if len(rows["index"]) != plan["dataset"]["frames"]:
            raise ValueError("Dataset size changed since submission")
        info, stats = read_json(root / "meta/info.json"), read_json(root / "meta/stats.json")
        for path in files:
            if set(plan["columns"].values()) & set(pq.read_schema(path).names):
                raise ValueError("Output columns already exist; use a new --name")
        columns = collect(plan, rows)
        if any(not np.isfinite(val).all() for val in columns.values()):
            raise ValueError("Values/advantages overflowed float32")
        print("Generating statistics, histograms, and episode plots", flush=True)
        result = report(plan, rows, columns)
        # Map by global row id, never assume parquet file ordering or contiguous episodes per shard.
        by_index = {key: vals[np.argsort(rows["index"])] for key, vals in columns.items()}
        for key, vals in columns.items():
            info["features"][key] = {"dtype": "float32", "shape": [1], "names": None}
            stats[key] = feature_stats(vals)
        # Stage every rewrite before replacing originals. Backup originals for recovery if commit is interrupted.
        with tempfile.TemporaryDirectory(prefix=".awr-stage-", dir=root) as temp:
            stage = Path(temp)
            replacements = []
            for path in files:
                print(f"Staging dataset update: {path.name}", flush=True)
                table = pq.read_table(path)
                indices = scalar_column(table, "index", np.int64)
                table = add_columns(table, {key: vals[indices] for key, vals in by_index.items()})
                dest = stage / path.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dest)
                replacements.append((dest, path))
            for relative, data in (("meta/stats.json", stats), ("meta/info.json", info)):
                dest = stage / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_json(dest, data)
                replacements.append((dest, root / relative))
            print("Backing up originals and committing dataset updates", flush=True)
            backup = root / "meta/awr_backups" / plan["name"]
            backup.mkdir(parents=True, exist_ok=False)
            for _, path in replacements:
                dest = backup / path.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, dest)
            for source, path in replacements:
                os.replace(source, path)
        write_json(
            Path(plan["output_dir"]) / "complete.json", {"columns": plan["columns"], "backup": str(backup)}
        )
        print(json.dumps(result, indent=2))
        print(f"Updated {root}\nHistograms: {plan['output_dir']}/histograms.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Validate inputs and write a plan for the Bash launcher"
    )
    prepare_parser.add_argument("dataset_root", type=Path)
    prepare_parser.add_argument("run_template")
    prepare_parser.add_argument("checkpoint")
    prepare_parser.add_argument("output_dir", type=Path)
    prepare_parser.add_argument("name", help="Column label; empty uses run family plus checkpoint")
    prepare_parser.add_argument("--dry-run", action="store_true")
    merge_parser = commands.add_parser("merge", help="Validate and merge a completed inference plan")
    merge_parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        write_plan(args)
    else:
        merge(args.plan)


if __name__ == "__main__":
    main()
