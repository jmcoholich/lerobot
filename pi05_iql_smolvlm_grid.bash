#!/bin/bash
#SBATCH --job-name=sparse_smolvlm_value_grid
#SBATCH --array=0-54%10
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --output=slurm-%A_%a.out
#SBATCH --exclude=ig-88,megazord,cyborg

set -euo pipefail

GRID_NAME=${1:-sparse_smolvlm_value_grid}
VALUE_KEY=${2:-sparse_returns_gamma_0.99}
TEST_DATASET=${3:-walle_skywalker_testset}

DROPOUTS=(0 10 20 30 40 50 60 70 80 90 100)
WEIGHT_DECAYS=(0.01 0.1 1 5 10)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
NUM_DROPOUTS=${#DROPOUTS[@]}
NUM_RUNS=$((NUM_DROPOUTS * ${#WEIGHT_DECAYS[@]}))

if ((TASK_ID < 0 || TASK_ID >= NUM_RUNS)); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((NUM_RUNS - 1)), got $TASK_ID" >&2
    exit 1
fi

DROPOUT=${DROPOUTS[$((TASK_ID % NUM_DROPOUTS))]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$((TASK_ID / NUM_DROPOUTS))]}
JOB_NAME="${GRID_NAME}_dropout${DROPOUT}_wd${WEIGHT_DECAY}"

REPO_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$REPO_DIR"

echo "Grid task: $TASK_ID/$((NUM_RUNS - 1))"
echo "Dropout: $DROPOUT"
echo "Weight decay: $WEIGHT_DECAY"
echo "Training job name: $JOB_NAME"

bash "$REPO_DIR/pi05_iql_train_value_fn.bash" \
    "$JOB_NAME" \
    "$VALUE_KEY" \
    smolvla \
    "$TEST_DATASET" \
    "$WEIGHT_DECAY" \
    false \
    "$DROPOUT"
