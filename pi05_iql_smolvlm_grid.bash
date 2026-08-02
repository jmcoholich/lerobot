#!/bin/bash
#SBATCH --job-name=smolvlm_lr_grid
#SBATCH --array=0-20%80
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --qos=long
#SBATCH --output=slurm-%A_%a.out
#SBATCH --exclude=ig-88,megazord,cyborg,sonny,spd-13

set -euo pipefail

GRID_NAME=${1:-smolvlm_lr_grid_256_bs}

LEARNING_RATES=(3e-5 1e-4 3e-4)
SEEDS=(0 1 2)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
NUM_SEEDS=${#SEEDS[@]}
NUM_RUNS=$((${#LEARNING_RATES[@]} * NUM_SEEDS))

if ((TASK_ID < 0 || TASK_ID >= NUM_RUNS)); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((NUM_RUNS - 1)), got $TASK_ID" >&2
    exit 1
fi

SEED=${SEEDS[$((TASK_ID % NUM_SEEDS))]}
LR=${LEARNING_RATES[$((TASK_ID / NUM_SEEDS))]}
JOB_NAME="${GRID_NAME}_lr${LR}"

REPO_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$REPO_DIR"

echo "Grid task: $TASK_ID/$((NUM_RUNS - 1))"
echo "Learning rate: $LR"
echo "Seed: $SEED"
echo "Training job name: ${JOB_NAME}_seed_${SEED}"

SEED="$SEED" \
LR="$LR" \
bash "$REPO_DIR/pi05_iql_train_value_fn.bash" "$JOB_NAME"
