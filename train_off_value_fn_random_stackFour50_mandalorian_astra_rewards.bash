#!/bin/bash

set -euo pipefail

DATASET=stackFour50_mandalorian_astra_rewards
VALUE_KEY=codex_astra_reward_v6_return_gamma_0.95
REWARD_KEY=codex_astra_reward_v6
WANDB_PROJECT=OOF_value_fn
NUM_EPISODES=100
JOB_PREFIX=${1:-off_value_fn_random_stackFour50_mandalorian_astra_rewards_frozen_dropout25}

# Generated once with random.Random(1000); shuffled range(100), split into five groups of 20.
TEST_FOLDS=(
    "3,6,11,15,19,26,34,37,48,63,65,66,70,72,73,76,87,89,91,93"
    "4,9,10,17,31,38,41,44,47,51,57,61,71,78,79,80,81,86,94,95"
    "1,7,14,24,33,35,39,40,42,43,49,52,56,60,67,69,75,88,96,98"
    "0,2,5,13,18,20,22,23,27,32,36,53,58,64,74,77,82,83,90,92"
    "8,12,16,21,25,28,29,30,45,46,50,54,55,59,62,68,84,85,97,99"
)

for fold in "${!TEST_FOLDS[@]}"; do
    test_episodes=${TEST_FOLDS[$fold]}
    IFS=, read -ra test_indices <<< "$test_episodes"
    unset test_set
    declare -A test_set=()
    for episode in "${test_indices[@]}"; do
        test_set[$episode]=1
    done

    train_episodes=""
    for ((episode = 0; episode < NUM_EPISODES; episode++)); do
        if [[ -z ${test_set[$episode]+x} ]]; then
            train_episodes+="${train_episodes:+,}$episode"
        fi
    done

    job_name="${JOB_PREFIX}_fold_$((fold + 1))"
    echo "Submitting $job_name with ${#test_indices[@]} random test episodes"
    DATASET="$DATASET" \
    VALUE_KEY="$VALUE_KEY" \
    REWARD_KEY="$REWARD_KEY" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    INPUT_DROPOUT_PERCENT=25 \
    FREEZE_VISION_ENCODER=true \
    TRAIN_EPISODES="$train_episodes" \
    TEST_EPISODES="$test_episodes" \
        sbatch train_MLP_value_function.bash "$job_name"
done
