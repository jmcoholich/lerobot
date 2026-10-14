#!/bin/bash

set -euo pipefail

DATASET=stackFour50_mandalorian_astra_rewards
EPISODES=$(seq -s, 0 99)
# No GPU node features are configured: request one GPU and exclude other types.
# Preserve the worker script's existing node exclusions as well.
excluded_nodes=$(sed -n 's/^#SBATCH --exclude=//p' pi05_value_inference_static.bash)
other_gpu_nodes=$(sinfo -p kira-lab -N -h -o '%N %G' |
    awk '$2 !~ /^gpu:(2080_ti|titan_x):/ {print $1}' | sort -u | paste -sd, -)
excluded_nodes+="${other_gpu_nodes:+${excluded_nodes:+,}$other_gpu_nodes}"
CONFIGS=(frozen_dropout25 unfrozen_dropout0)

for config in "${CONFIGS[@]}"; do
    for fold in {1..5}; do
        job_name="off_value_fn_random_${DATASET}_${config}_fold_${fold}"
        echo "Submitting inference for $job_name"
        submission=$(sbatch --parsable \
            --gpus=1 \
            --exclude="$excluded_nodes" \
            --array="$EPISODES" \
            pi05_value_inference_static.bash \
            "$job_name" \
            "$DATASET" \
            last \
            "$EPISODES")
        job_id=${submission%%;*}
        echo "Submitted Slurm array $job_id"
    done
done
