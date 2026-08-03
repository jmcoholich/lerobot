#!/usr/bin/env python

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from accelerate import Accelerator

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import ensure_raw_pi05_scalar_preprocessor, evaluate_scalar_predictor
from lerobot.utils.random_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the saved PI0.5 model's test-set loss.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--output-file", default="test_loss.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available")

    policy_path = Path(args.policy_path)
    cfg = TrainPipelineConfig.from_pretrained(policy_path)
    if cfg.test_dataset is None:
        raise ValueError(f"No test dataset is recorded in {policy_path / 'train_config.json'}")

    cfg.policy.pretrained_path = policy_path
    cfg.policy.device = "cuda"
    cfg.policy.gradient_checkpointing = False

    accelerator = Accelerator()
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    train_dataset = make_dataset(cfg)
    test_dataset = make_dataset(dataclasses.replace(cfg, dataset=cfg.test_dataset))
    policy = make_policy(cfg.policy, ds_meta=train_dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": accelerator.device.type}},
    )
    ensure_raw_pi05_scalar_preprocessor(preprocessor)

    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            test_dataset,
            range(0, len(test_dataset), cfg.test_frame_stride),
        ),
        num_workers=cfg.num_workers,
        batch_size=cfg.test_batch_size,
        pin_memory=accelerator.device.type == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    policy, test_dataloader = accelerator.prepare(policy, test_dataloader)
    metrics = evaluate_scalar_predictor(
        policy,
        test_dataloader,
        preprocessor,
        accelerator,
        train_dataset.meta.stats,
    )

    if accelerator.is_main_process:
        output_file = Path(args.output_file)
        output_file.write_text(f"{metrics['loss']}\n", encoding="utf-8")
        print(f"Test loss: {metrics['loss']}")
        print(f"Wrote test loss to {output_file.resolve()}")


if __name__ == "__main__":
    main()
