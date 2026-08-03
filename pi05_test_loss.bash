#!/bin/bash
#SBATCH --job-name=pi05_test_loss
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --qos=short
#SBATCH --exclude=ig-88,megazord,cyborg,megazord,sonny,spd-13

echo "Hostname: $(hostname)"
if ! GPU_STATUS=$(nvidia-smi 2>&1) || [[ "$GPU_STATUS" == *ERR* ]]; then
    echo "GPU is not available:" >&2
    echo "$GPU_STATUS" >&2
    exit 1
fi

POLICY_NAME=${1:?Pass the policy name as the first argument}
CHECKPOINT_NAME=${2:?Pass the checkpoint name as the second argument}
MODEL_PATH="outputs/$POLICY_NAME/checkpoints/$CHECKPOINT_NAME/pretrained_model"
OUTPUT_FILE="${POLICY_NAME}_${CHECKPOINT_NAME}_test_loss.txt"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot
cd /coc/testnvme/$USER/lerobot_iql
export PYTHONPATH="$PWD/src:${PYTHONPATH}"

echo "Hostname: $(hostname)"
if ! GPU_STATUS=$(nvidia-smi 2>&1) || [[ "$GPU_STATUS" == *ERR* ]]; then
    echo "GPU is not available:" >&2
    echo "$GPU_STATUS" >&2
    exit 1
fi

python src/lerobot/scripts/lerobot_pi05_test_loss.py \
    --policy-path="$MODEL_PATH" \
    --output-file="$OUTPUT_FILE"