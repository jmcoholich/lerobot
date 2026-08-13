#!/bin/bash
#SBATCH --job-name=value_fn_annotated_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-279%40
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --qos=short
#SBATCH --exclude=ig-88,megazord,cyborg,megazord,sonny,spd-13

set -euo pipefail

# Build 144 balanced configurations from the important axes. For each combination,
# sample two distinct (learning rate, tau, MLP dropout) settings, shuffle with a
# fixed seed, and keep 140. Expand each configuration to two seeds and shuffle
# again so neighboring array tasks do not run similar configurations.
read -r CONFIG_ID FREEZE_VISION_ENCODER INPUT_DROPOUT_PERCENT DISCOUNT LR N_STEP TAU MLP_DROPOUT SEED < <(
    python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import itertools
import random
import sys

task_id = int(sys.argv[1])
rng = random.Random(20260813)

core_configs = itertools.product(
    ("false", "true"),       # trainable vs frozen vision encoder
    (0, 25, 50, 100),        # input dropout percent
    (0.9, 0.95, 0.99),       # return/bootstrapping discount
    (0, 1, 10),              # bootstrapping horizon
)
secondary_configs = list(itertools.product(
    ("3e-6", "1e-5", "3e-5"),
    (0.005, 0.05),
    (0.0, 0.1),
))

configs = []
for freeze, input_dropout, discount, n_step in core_configs:
    for lr, tau, mlp_dropout in rng.sample(secondary_configs, 2):
        configs.append((freeze, input_dropout, discount, lr, n_step, tau, mlp_dropout))

rng.shuffle(configs)
configs = configs[:140]
jobs = [
    (config_id, *config, seed)
    for config_id, config in enumerate(configs)
    for seed in (1000, 2000)
]
rng.shuffle(jobs)

if not 0 <= task_id < len(jobs):
    raise SystemExit(f"Array task {task_id} is outside 0-{len(jobs) - 1}")
print(*jobs[task_id])
PY
)

if [ "$FREEZE_VISION_ENCODER" = "true" ]; then
    FREEZE_LABEL=frozen
else
    FREEZE_LABEL=trainable
fi
JOB_NAME="annotated_mlp_cfg${CONFIG_ID}_${FREEZE_LABEL}_seed${SEED}"

echo "Sweep array task: $SLURM_ARRAY_TASK_ID"
echo "Sweep configuration: $CONFIG_ID"
echo "Job name: $JOB_NAME"
echo "Freeze vision encoder: $FREEZE_VISION_ENCODER"
echo "Input dropout percent: $INPUT_DROPOUT_PERCENT"
echo "Discount: $DISCOUNT"
echo "Learning rate: $LR"
echo "N-step horizon: $N_STEP"
echo "Target-network tau: $TAU"
echo "MLP dropout: $MLP_DROPOUT"
echo "Seed: $SEED"

FREEZE_VISION_ENCODER="$FREEZE_VISION_ENCODER" \
INPUT_DROPOUT_PERCENT="$INPUT_DROPOUT_PERCENT" \
DISCOUNT="$DISCOUNT" \
LR="$LR" \
N_STEP="$N_STEP" \
TAU="$TAU" \
MLP_DROPOUT="$MLP_DROPOUT" \
SEED="$SEED" \
bash train_MLP_value_function.bash "$JOB_NAME"
