#!/bin/bash

set -euo pipefail

DATASET=plug5_offline_rl_dataset_walle_skywalker_testset_annotated
WANDB_PROJECT=OOF_value_fn
NUM_EPISODES=388
JOB_PREFIX=${1:-OOF_value_fn}
TEST_RANGES=(0:78 78:156 156:234 234:311 311:388)

for fold in "${!TEST_RANGES[@]}"; do
    test_range=${TEST_RANGES[$fold]}
    IFS=: read -r fold_start fold_end <<< "$test_range"

    train_episodes=""
    for ((episode = 0; episode < NUM_EPISODES; episode++)); do
        if ((episode < fold_start || episode >= fold_end)); then
            train_episodes+="${train_episodes:+,}$episode"
        fi
    done

    job_name="${JOB_PREFIX}_fold_$((fold + 1))"
    echo "Submitting $job_name with test episodes $test_range"
    DATASET="$DATASET" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    TRAIN_EPISODES="$train_episodes" \
    TEST_EPISODES="$test_range" \
        sbatch train_MLP_value_function.bash "$job_name"
done
