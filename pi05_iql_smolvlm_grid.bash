#!/bin/bash
#SBATCH --job-name=bootstrap_100_grid
#SBATCH --array=0-164%80
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --output=slurm-%A_%a.out
#SBATCH --exclude=ig-88,megazord,cyborg,sonny,spd-13

set -euo pipefail

GRID_NAME=${1:-bootstrap_100}

DROPOUTS=(0 10 20 30 40 50 60 70 80 90 100)
WEIGHT_DECAYS=(0.01 0.1 1 5 10)
SEEDS=(0 1 2)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
NUM_DROPOUTS=${#DROPOUTS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_RUNS=$((NUM_DROPOUTS * ${#WEIGHT_DECAYS[@]} * NUM_SEEDS))

if ((TASK_ID < 0 || TASK_ID >= NUM_RUNS)); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((NUM_RUNS - 1)), got $TASK_ID" >&2
    exit 1
fi

SEED=${SEEDS[$((TASK_ID % NUM_SEEDS))]}
COMBINATION_ID=$((TASK_ID / NUM_SEEDS))
DROPOUT=${DROPOUTS[$((COMBINATION_ID % NUM_DROPOUTS))]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$((COMBINATION_ID / NUM_DROPOUTS))]}
JOB_NAME="${GRID_NAME}_dropout${DROPOUT}_wd${WEIGHT_DECAY}"

REPO_DIR=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$REPO_DIR"

echo "Grid task: $TASK_ID/$((NUM_RUNS - 1))"
echo "Dropout: $DROPOUT"
echo "Weight decay: $WEIGHT_DECAY"
echo "Seed: $SEED"
echo "Training job name: ${JOB_NAME}_seed_${SEED}"

SEED="$SEED" \
N_STEP=100 \
WEIGHT_DECAY="$WEIGHT_DECAY" \
INPUT_DROPOUT_PERCENT="$DROPOUT" \
bash "$REPO_DIR/pi05_iql_train_value_fn.bash" "$JOB_NAME"
