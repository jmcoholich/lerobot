#!/bin/bash
# Resource settings and conda activation are in pi05_training.bash.
set -euo pipefail

TRAINING_DIR="/coc/testnvme/jcoholich3/lerobot"
DATASETS=(
    "plug5_offline_rl_dataset_walle_skywalker_testset_annotated"
    "stackFour50_mandalorian_astra_rewards"
)
ADVANTAGE_KEYS=(
    "awr_advantage_off_value_fn_random_unfrozen_dropout0_last_OOF"
    "awr_advantage_off_value_fn_random_stackFour50_mandalorian_astra_rewards_frozen_dropout25_last_OOF"
)
DATASET_LABELS=(plug5 stack4)
JOB_PREFIX="${1:-awr}"
TEMPERATURES=(0.5 1 2 3 4)
# Exclude the untransformed baseline and mutually exclusive std + norm combinations.
MODES=(bin std norm std_bin norm_bin)

cd "$TRAINING_DIR"
for i in "${!DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        flags=()
        case "$mode" in
            std*) flags+=(--standardize_advantages) ;;
            norm*) flags+=(--normalize_advantages) ;;
        esac
        if [[ "$mode" == *bin ]]; then
            flags+=(--binarize_advantages)
        fi
        for temperature in "${TEMPERATURES[@]}"; do
            job_name="${JOB_PREFIX}_${DATASET_LABELS[$i]}_${mode}_t${temperature}"
            echo "Submitting $job_name"
            sbatch --job-name="$job_name" pi05_training.bash \
                "${DATASETS[$i]}" "$job_name" "$temperature" "${ADVANTAGE_KEYS[$i]}" "${flags[@]}"
        done
    done
done
