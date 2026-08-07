#!/usr/bin/env python

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import pyarrow.dataset as pa_ds
import torch
from accelerate import Accelerator
from tqdm import tqdm

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import ensure_raw_pi05_scalar_preprocessor, evaluate_scalar_predictor
from lerobot.utils.random_utils import set_seed

DEFAULT_CONSTANT_SCALAR = 0.15862231207103095


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the saved PI0.5 model's test-set loss.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument(
        "--constant-scalar",
        type=float,
        nargs="?",
        const=DEFAULT_CONSTANT_SCALAR,
        help=f"Skip model loading and predict a constant (default: {DEFAULT_CONSTANT_SCALAR}).",
    )
    parser.add_argument("--output-file", default="test_loss.txt")
    return parser.parse_args()


def evaluate_constant_predictor(cfg, scalar: float, accelerator: Accelerator) -> dict:
    dataset_root = cfg.test_dataset.root
    if dataset_root is None:
        raise ValueError("Constant evaluation requires test_dataset.root in train_config.json")

    dataset = pa_ds.dataset(Path(dataset_root) / "data", format="parquet")
    row_filter = None
    if cfg.test_dataset.episodes is not None:
        row_filter = pa_ds.field("episode_index").isin(cfg.test_dataset.episodes)
    table = dataset.to_table(columns=[cfg.policy.value_key], filter=row_filter)
    targets = torch.tensor(table.column(0).to_numpy(), dtype=torch.float32)[
        :: cfg.test_frame_stride
    ]
    dataloader = accelerator.prepare(
        torch.utils.data.DataLoader(targets, batch_size=cfg.test_batch_size)
    )

    total_squared_error = 0.0
    num_samples = 0
    start_time = time.perf_counter()
    test_batches = tqdm(
        dataloader,
        desc="Evaluating test batches",
        unit="batch",
        file=sys.stdout,
        disable=not accelerator.is_local_main_process,
    )
    for batch in test_batches:
        squared_error = accelerator.gather_for_metrics((batch.float() - scalar).square())
        total_squared_error += squared_error.sum().item()
        num_samples += squared_error.numel()
    torch.cuda.synchronize()

    return {
        "loss": total_squared_error / num_samples,
        "num_samples": num_samples,
        "eval_s": time.perf_counter() - start_time,
    }


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

    if args.constant_scalar is not None:
        metrics = evaluate_constant_predictor(cfg, args.constant_scalar, accelerator)
    else:
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
        test_batches = tqdm(
            test_dataloader,
            desc="Evaluating test batches",
            unit="batch",
            file=sys.stdout,
            disable=not accelerator.is_local_main_process,
        )
        metrics = evaluate_scalar_predictor(
            policy,
            test_batches,
            preprocessor,
            accelerator,
            train_dataset.meta.stats,
        )

    if accelerator.is_main_process:
        output_file = Path(args.output_file)
        output_file.write_text(
            (f"Constant prediction: {args.constant_scalar}\n" if args.constant_scalar is not None else "")
            + f"Test loss: {metrics['loss']}\nEval time (seconds): {metrics['eval_s']}\n",
            encoding="utf-8",
        )
        if args.constant_scalar is not None:
            print(f"Constant prediction: {args.constant_scalar}")
        print(f"Test loss: {metrics['loss']}")
        print(f"Eval time (seconds): {metrics['eval_s']}")
        print(f"Wrote test loss to {output_file.resolve()}")


if __name__ == "__main__":
    main()
