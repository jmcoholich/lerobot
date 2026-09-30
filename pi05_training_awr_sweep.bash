#!/bin/bash
# Resource settings and conda activation are in pi05_training.bash.
set -euo pipefail

TRAINING_DIR="/coc/testnvme/jcoholich3/lerobot"
DATASET="plug5_offline_rl_dataset_walle_skywalker_testset_annotated"
ADVANTAGE_KEY="awr_advantage_off_value_fn_random_unfrozen_dropout0_last_OOF"
JOB_PREFIX="${1:-pi05_awr_oof}"
# Weights are min(exp(T * advantage), 20); observed maxima are about 1.7, 2.9, 8.6, 20, and 20.
TEMPERATURES=(0.5 1 2 3 4)

cd "$TRAINING_DIR"
for temperature in "${TEMPERATURES[@]}"; do
    job_name="${JOB_PREFIX}_temp_${temperature}"
    echo "Submitting $job_name"
    sbatch --job-name="$job_name" pi05_training.bash \
        "$DATASET" "$job_name" "$temperature" "$ADVANTAGE_KEY"
done
