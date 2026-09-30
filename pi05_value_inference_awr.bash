#!/bin/bash
# Edit these settings or override them through the environment.
# Slurm resources live in pi05_value_inference_static.bash and pi05_value_merge_awr.bash.
set -eo pipefail
source "/coc/testnvme/$USER/.bashrc"
conda activate lerobot
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_TEMPLATE=${RUN_TEMPLATE:-'off_value_fn_random_unfrozen_dropout0_fold_{fold}'}
CHECKPOINT=${CHECKPOINT:-last}
DATASET_ROOT=${DATASET_ROOT:-"/coc/testnvme/jcoholich3/lerobot_data/plug5_offline_rl_dataset_walle_skywalker_testset_annotated"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/awr"}
NAME=${NAME:-}
MAX_PARALLEL=${MAX_PARALLEL:-} # Empty means no array concurrency cap.

# Python reads the checkpoint splits and validates the return targets before any jobs are submitted.
output=$(python src/lerobot/scripts/lerobot_pi05_value_awr.py prepare \
    "$DATASET_ROOT" "$RUN_TEMPLATE" "$CHECKPOINT" "$OUTPUT_DIR" "$NAME" "$@")
if (( $# )); then
    echo "$output"
    exit 0
fi

echo "Report directory: $output"
mkdir -p slurm_logs
job_ids=""
while IFS=$'\t' read -r run policy_path manifest episodes; do
    submission=$(POLICY_PATH="$policy_path" DATASET_ROOT="$DATASET_ROOT" \
        INFERENCE_RUN_NAME="$manifest" INFERENCE_OUTPUT_DIR="$output" \
        SKIP_VIDEO=1 SKIP_MANIFEST=1 \
        sbatch --parsable --export=ALL \
            --array="${episodes}${MAX_PARALLEL:+%$MAX_PARALLEL}" --job-name="awr_$manifest" \
            pi05_value_inference_static.bash \
            "$run" "$(basename "$DATASET_ROOT")" "$CHECKPOINT" "$episodes")
    job_id=${submission%%;*}
    printf '%s\t%s\n' "$manifest" "$job_id" >> "$output/jobs.tsv"
    echo "Submitted $manifest: $job_id"
    job_ids+="${job_ids:+:}$job_id"
done < "$output/folds.tsv"

submission=$(sbatch --parsable --dependency="afterok:$job_ids" \
    pi05_value_merge_awr.bash "$output/plan.json")
job_id=${submission%%;*}
printf 'merge\t%s\n' "$job_id" >> "$output/jobs.tsv"
echo "Submitted merge: $job_id"
