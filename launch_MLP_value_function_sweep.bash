#!/bin/bash

set -euo pipefail

SWEEP_NAME=${1:-mlp_sweep}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_MLP_value_function.bash}
DRY_RUN=${DRY_RUN:-false}
SEEDS=(0 1 2)
CONFIG_COUNT=0

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

submit_config() {
    local suffix=$1
    local freeze_vision_encoder=$2
    local learning_rate=$3
    local input_dropout=$4
    local projection_dim=$5
    local hidden_dim=$6
    local mlp_dropout=$7
    local config_id
    local run_name
    local -a command

    printf -v config_id "%02d" "$CONFIG_COUNT"
    run_name="${SWEEP_NAME}_${config_id}_${suffix}"

    for seed in "${SEEDS[@]}"; do
        command=(
            sbatch
            --job-name="${run_name}_seed_${seed}"
            --export="ALL,SEED=$seed,INIT=pi05,N_STEP=0,WEIGHT_DECAY=0.01,FREEZE_VISION_ENCODER=$freeze_vision_encoder,LR=$learning_rate,INPUT_DROPOUT_PERCENT=$input_dropout,VISION_PROJECTION_DIM=$projection_dim,MLP_HIDDEN_DIM=$hidden_dim,MLP_DROPOUT=$mlp_dropout"
            "$TRAIN_SCRIPT"
            "$run_name"
        )
        if [ "$DRY_RUN" = "true" ]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            "${command[@]}"
        fi
    done

    CONFIG_COUNT=$((CONFIG_COUNT + 1))
}

# 0: Current launcher defaults.
submit_config default true 1e-5 50 256 512 0.1

# 1-12: Frozen-encoder grid.
for learning_rate in 3e-5 1e-4; do
    for input_dropout in 0 25; do
        for projection_dim in 128 256 512; do
            suffix="frozen_lr${learning_rate}_input${input_dropout}_proj${projection_dim}"
            submit_config "$suffix" true "$learning_rate" "$input_dropout" "$projection_dim" 512 0.1
        done
    done
done

# 13-16: MLP capacity and dropout around the frozen baseline.
submit_config hidden256 true 1e-4 25 256 256 0.1
submit_config hidden1024 true 1e-4 25 256 1024 0.1
submit_config mlp_dropout0 true 1e-4 25 256 512 0.0
submit_config mlp_dropout03 true 1e-4 25 256 512 0.3

# 17-19: End-to-end vision fine-tuning.
submit_config finetune_lr3e-6_input25 false 3e-6 25 256 512 0.1
submit_config finetune_lr1e-5_input25 false 1e-5 25 256 512 0.1
submit_config finetune_lr1e-5_input0 false 1e-5 0 256 512 0.1

if [ "$CONFIG_COUNT" -ne 20 ]; then
    echo "Expected 20 configurations, generated $CONFIG_COUNT" >&2
    exit 1
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "Generated $((CONFIG_COUNT * ${#SEEDS[@]})) dry-run commands across $CONFIG_COUNT configurations."
else
    echo "Submitted $((CONFIG_COUNT * ${#SEEDS[@]})) runs across $CONFIG_COUNT configurations."
fi
