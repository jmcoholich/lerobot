#!/bin/bash

set -euo pipefail

EPISODES=1,5,45,50,51,52,53,54,55,56,57,58,59,7
dependency=()

for fold in {1..5}; do
    job_name="off_value_fn_random_fold_${fold}"
    echo "Submitting inference for $job_name"
    submission=$(sbatch --parsable \
        "${dependency[@]}" \
        --array="$EPISODES" \
        pi05_value_inference_static.bash \
        "$job_name" \
        plug5_offline_rl_dataset_annotated \
        last \
        "$EPISODES")
    job_id=${submission%%;*}
    dependency=(--dependency="afterok:$job_id")
    echo "Submitted Slurm array $job_id"
done
