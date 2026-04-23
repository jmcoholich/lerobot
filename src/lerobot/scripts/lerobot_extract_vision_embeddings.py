#!/usr/bin/env python

import argparse
import contextlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info
from lerobot.policies.factory import make_policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PI0.5 vision embeddings into a LeRobot dataset.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.root", dest="dataset_root", required=True)
    parser.add_argument("--dataset.revision", dest="dataset_revision", default=None)
    parser.add_argument("--dataset.video_backend", dest="dataset_video_backend", default=None)
    parser.add_argument("--policy.type", dest="policy_type", default="pi05")
    parser.add_argument("--policy.pretrained_path", dest="policy_pretrained_path", required=True)
    parser.add_argument("--policy.device", dest="policy_device", default="cuda")
    parser.add_argument("--policy.dtype", dest="policy_dtype", default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def _make_embedding_feature_name(camera_key: str) -> str:
    return f"embedding_tokens.{camera_key}"


def _make_probe_feature_name(camera_key: str) -> str:
    return f"embedding_probe.{camera_key}"


def _extract_embeddings(
    policy,
    batch: dict[str, torch.Tensor],
    camera_keys: list[str],
) -> dict[str, torch.Tensor]:
    images, _ = policy._preprocess_images(batch)
    embeddings = {}
    vision_tower = policy.model.paligemma_with_expert.paligemma.vision_tower
    projector = policy.model.paligemma_with_expert.paligemma.multi_modal_projector

    for camera_key, image in zip(camera_keys, images[: len(camera_keys)], strict=True):
        vision_outputs = vision_tower(image)
        image_tokens = projector(vision_outputs.last_hidden_state)
        probe_embedding = vision_outputs.pooler_output

        if image_tokens.ndim != 3:
            raise ValueError(f"Unexpected token shape for {camera_key}: {tuple(image_tokens.shape)}")
        if probe_embedding is None or probe_embedding.ndim != 2:
            raise ValueError(f"Unexpected probe shape for {camera_key}: {tuple(probe_embedding.shape)}")

        embeddings[_make_embedding_feature_name(camera_key)] = image_tokens.to(dtype=torch.float32).cpu()
        embeddings[_make_probe_feature_name(camera_key)] = probe_embedding.to(dtype=torch.float32).cpu()

    return embeddings


def main() -> None:
    args = parse_args()

    if args.policy_type != "pi05":
        raise ValueError(f"Only --policy.type=pi05 is supported, got {args.policy_type}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
        video_backend=args.dataset_video_backend,
    )
    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError("Dataset does not contain any camera keys.")

    policy_cfg = PI05Config(
        pretrained_path=Path(args.policy_pretrained_path),
        device=args.policy_device,
        dtype=args.policy_dtype,
        compile_model=False,
    )
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    token_shape = None
    probe_dim = None
    for data_path in sorted((dataset.root / "data").glob("*/*.parquet")):
        logging.info("Processing %s", data_path)
        df = pd.read_parquet(data_path).reset_index(drop=True)
        frame_indices = df["index"].astype(int).tolist()
        subset = Subset(dataset, frame_indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.policy_device.startswith("cuda"),
        )

        file_embeddings = {
            feature_name: []
            for camera_key in camera_keys
            for feature_name in (_make_embedding_feature_name(camera_key), _make_probe_feature_name(camera_key))
        }
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.policy_device.startswith("cuda")
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_context:
            for batch in tqdm(loader, desc=str(data_path.relative_to(dataset.root))):
                batch_embeddings = _extract_embeddings(policy, batch, camera_keys)
                for feature_name, values in batch_embeddings.items():
                    file_embeddings[feature_name].append(values.numpy())
                    if token_shape is None and feature_name.startswith("embedding_tokens."):
                        token_shape = tuple(int(dim) for dim in values.shape[1:])
                    if probe_dim is None and feature_name.startswith("embedding_probe."):
                        probe_dim = int(values.shape[-1])
                    if token_shape is not None and probe_dim is not None:
                        for camera_key in camera_keys:
                            dataset.meta.info["features"][_make_embedding_feature_name(camera_key)] = {
                                "dtype": "float32",
                                "shape": token_shape,
                                "names": None,
                            }
                            dataset.meta.info["features"][_make_probe_feature_name(camera_key)] = {
                                "dtype": "float32",
                                "shape": (probe_dim,),
                                "names": None,
                            }

        for feature_name, values in file_embeddings.items():
            df[feature_name] = list(np.concatenate(values, axis=0))

        tmp_path = data_path.with_suffix(".tmp.parquet")
        _write_parquet(df, tmp_path, dataset.meta)
        tmp_path.replace(data_path)

    if token_shape is None or probe_dim is None:
        raise ValueError("No embeddings were extracted.")

    write_info(dataset.meta.info, dataset.meta.root)
    logging.info("Added embedding features for cameras: %s", ", ".join(camera_keys))


if __name__ == "__main__":
    main()
