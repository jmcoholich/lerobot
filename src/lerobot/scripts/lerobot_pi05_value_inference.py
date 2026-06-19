#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.dataset as pa_ds
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Run PI0.5 value inference on a LeRobot dataset.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--output-dir", default="/coc/testnvme/jcoholich3/reward-modeling/viewer_files")
    parser.add_argument("--manifest-name", default="pi05_value.json")
    parser.add_argument("--video-prefix")
    parser.add_argument("--episodes", help="Comma-separated episode indices. Defaults to all episodes.")
    parser.add_argument("--num-frames", type=int, help="Frames to sample per episode. Defaults to all frames.")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    return parser.parse_args()


def episode_bounds(dataset, episode_index: int) -> tuple[int, int]:
    ep = dataset.meta.episodes[episode_index]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def episode_task(dataset, episode_index: int) -> tuple[int, str]:
    ep = dataset.meta.episodes[episode_index]
    task = ep["tasks"][0]
    task_index = int(dataset.meta.tasks.loc[task]["task_index"])
    return task_index, str(task)


def sample_indices(start: int, end: int, n: int | None) -> tuple[list[int], list[int]]:
    if n is None:
        rel = list(range(end - start))
    else:
        rel = np.linspace(0, end - start - 1, min(n, end - start), dtype=int).tolist()
    rel = list(dict.fromkeys(rel))
    return [start + i for i in rel], rel


def frame_to_rgb(frame) -> np.ndarray:
    frame = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(frame, 3, axis=-1) if frame.shape[-1] == 1 else frame


def write_all_cams_video(dataset, camera_keys: list[str], start: int, end: int, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = dataset[start]
    height = min(frame_to_rgb(first[key]).shape[0] for key in camera_keys)

    def concat(idx: int):
        item = dataset[idx]
        frames = []
        for key in camera_keys:
            frame = frame_to_rgb(item[key])
            width = round(frame.shape[1] * height / frame.shape[0])
            frames.append(cv2.resize(frame, (width, height)))
        frame = np.concatenate(frames, axis=1)
        return frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]

    with av.open(str(path), mode="w") as container:
        frame = concat(start)
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = frame.shape[1], frame.shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"movflags": "+faststart"}
        for idx in range(start, end):
            for packet in stream.encode(av.VideoFrame.from_ndarray(concat(idx), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def to_float_list(values) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1).tolist()


def norm_mode_name(norm_mode) -> str:
    return getattr(norm_mode, "value", str(norm_mode)).split(".")[-1].upper()


def value_norm_from_preprocessor(preprocessor, value_key: str) -> tuple[str | None, dict]:
    for step in getattr(preprocessor, "steps", []):
        normalized_keys = getattr(step, "normalize_complementary_data_keys", None)
        if normalized_keys is None or value_key not in normalized_keys:
            continue
        tensor_stats = getattr(step, "_tensor_stats", {})
        if value_key not in tensor_stats:
            raise KeyError(f"Checkpoint normalizes '{value_key}' but does not contain its stats")
        return norm_mode_name(getattr(step, "complementary_data_norm_mode", "QUANTILES")), tensor_stats
    return None, {}


def unnormalize_values(values: torch.Tensor, value_key: str, value_stats: dict, norm_mode: str | None):
    if norm_mode is None:
        return values
    if value_key not in value_stats:
        raise KeyError(f"Cannot unnormalize predictions: missing stats for '{value_key}'")

    stats = value_stats[value_key]
    norm_mode = str(norm_mode).upper()

    def stat_tensor(name: str):
        if name not in stats:
            raise KeyError(f"Cannot unnormalize predictions: missing '{name}' stats for '{value_key}'")
        return torch.as_tensor(stats[name], dtype=values.dtype, device=values.device)

    if norm_mode == "MEAN_STD":
        return values * stat_tensor("std") + stat_tensor("mean")
    if norm_mode == "MIN_MAX":
        return (values + 1.0) * (stat_tensor("max") - stat_tensor("min")) / 2.0 + stat_tensor("min")
    if norm_mode == "QUANTILES":
        return (values + 1.0) * (stat_tensor("q99") - stat_tensor("q01")) / 2.0 + stat_tensor("q01")
    if norm_mode == "QUANTILE10":
        return (values + 1.0) * (stat_tensor("q90") - stat_tensor("q10")) / 2.0 + stat_tensor("q10")

    raise ValueError(f"Unsupported complementary data normalization mode: {norm_mode}")


def episode_targets(dataset_root: Path, episode_index: int, value_key: str) -> dict[int, float]:
    table = pa_ds.dataset(dataset_root / "data", format="parquet").to_table(
        columns=["frame_index", value_key],
        filter=pa_ds.field("episode_index") == episode_index,
    )
    data = table.to_pydict()
    return {int(frame): float(value) for frame, value in zip(data["frame_index"], data[value_key])}


def predict_values(
    dataset,
    indices: list[int],
    policy,
    preprocessor,
    batch_size: int,
    value_key: str,
    value_stats: dict,
    value_norm_mode: str | None,
):
    values = []
    policy.reset()
    preprocessor.reset()
    for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False):
        with torch.inference_mode():
            pred = policy.predict_values(preprocessor(batch)).detach().float().cpu()
            pred = unnormalize_values(pred, value_key, value_stats, value_norm_mode)
        values.extend(to_float_list(pred))
    return values


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    policy_path = Path(args.policy_path)
    output_dir = Path(args.output_dir)
    repo_id = args.repo_id or dataset_root.name
    manifest_stem = args.manifest_name.removesuffix(".json")
    video_prefix = args.video_prefix or f"output_videos/{manifest_stem}"
    video_root = output_dir.parent / video_prefix

    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = "cuda"
    cfg.gradient_checkpointing = False

    policy = make_policy(cfg, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    value_norm_mode, value_stats = value_norm_from_preprocessor(preprocessor, cfg.value_key)

    key = f"{policy_path.parents[2].name}_{policy_path.parent.name}"
    camera_keys = list(dataset.meta.camera_keys)
    episodes = (
        [int(ep) for ep in args.episodes.split(",")]
        if args.episodes
        else list(range(int(dataset.meta.total_episodes)))
    )

    manifest_entries = []
    for episode_index in episodes:
        start, end = episode_bounds(dataset, episode_index)
        global_indices, frame_indices = sample_indices(start, end, args.num_frames)
        task_index, instruction = episode_task(dataset, episode_index)
        values = predict_values(
            dataset,
            global_indices,
            policy,
            preprocessor,
            args.batch_size,
            cfg.value_key,
            value_stats,
            value_norm_mode,
        )
        targets_by_frame = episode_targets(dataset_root, episode_index, cfg.value_key)
        targets = [targets_by_frame[frame] for frame in frame_indices]
        advantages = [target - value for target, value in zip(targets, values)]

        base = f"{manifest_stem}/task_{task_index}/episode_{episode_index}/all_cams"
        video_rel = f"{video_prefix}/task_{task_index}/episode_{episode_index}/all_cams.mp4"
        if not args.skip_video:
            write_all_cams_video(
                dataset,
                camera_keys,
                start,
                end,
                video_root / f"task_{task_index}/episode_{episode_index}/all_cams.mp4",
                float(dataset.fps),
            )

        json_path = output_dir / f"{base}.json"
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = {}

        result.update({
            "instruction": instruction,
            "video": video_rel,
            "num_frames": len(values),
            "video_metadata": {
                "fps": float(dataset.fps),
                "total_frames": end - start,
                "duration_seconds": (end - start) / float(dataset.fps),
            },
            key: {
                "model": str(policy_path),
                "value_key": cfg.value_key,
                "prediction_scale": "raw" if value_norm_mode is not None else "model_output",
                "unnormalized_from": value_norm_mode,
                "camera_keys": camera_keys,
                "values": values,
                "progress_scores": values,
                "frame_indices": frame_indices,
            },
            cfg.value_key: {
                "source": str(dataset_root),
                "value_key": cfg.value_key,
                "values": targets,
                "progress_scores": targets,
                "frame_indices": frame_indices,
            },
            f"advantages_{key}_{cfg.value_key}": {
                "value_key": cfg.value_key,
                "prediction_key": key,
                "values": advantages,
                "progress_scores": advantages,
                "frame_indices": frame_indices,
            },
        })

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        manifest_entries.append(base)
        print(f"Wrote {json_path}")

    if not args.skip_manifest:
        with (output_dir / args.manifest_name).open("w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2)


if __name__ == "__main__":
    main()
