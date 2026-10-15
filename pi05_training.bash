#!/bin/bash
#SBATCH --job-name=pi05_training
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH -x nestor,chappie,ig-88,perseverance,sonny,voltron

# Usage: sbatch pi05_training.bash DATASET JOB_NAME TEMPERATURE ADVANTAGE_KEY [FLAGS...]
# Supply all four positional arguments before any flags; a non-null temperature enables AWR.
# Example: sbatch pi05_training.bash my_dataset my_job 2 awr_advantage \
#     --normalize_advantages --binarize_advantages
# Flags take no values and are disabled by default:
#   --standardize_advantages: (advantage - dataset mean) / dataset std.
#   --normalize_advantages: (advantage - dataset median) / (dataset p99 - dataset p1).
#   --binarize_advantages: positive -> 1, negative -> -1, zero -> 0.
# Standardization and normalization are mutually exclusive and use dataset metadata statistics.
# Binarization runs after either scaling operation; all transforms precede weight conversion.

JOB_NAME=$2
OUTDIR=./outputs/$JOB_NAME
CHUNK=40
LR=5e-5
DATASET=$1
TEMPERATURE=${3:-null}
ADVANTAGE_KEY=${4:-awr_advantage}
PI05_BASE_PRETRAINED_PATH=/coc/testnvme/jcoholich3/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
echo "AWR temperature: $TEMPERATURE"
echo "AWR advantage key: $ADVANTAGE_KEY"
echo "PI05 pretrained path: $PI05_BASE_PRETRAINED_PATH"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

# Isolate dataset cache locks per job on local storage.
HF_DATASETS_CACHE=$(mktemp -d /tmp/lerobot-training-cache.XXXXXX) || exit 1
export HF_DATASETS_CACHE
trap 'rm -rf -- "$HF_DATASETS_CACHE"' EXIT

python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=$DATASET \
    --dataset.root="/coc/testnvme/jcoholich3/lerobot_data/$DATASET" \
    --policy.type=pi05 \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    --policy.pretrained_path="$PI05_BASE_PRETRAINED_PATH" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.awr_temperature=$TEMPERATURE \
    --policy.awr_advantage_key="$ADVANTAGE_KEY" \
    --policy.chunk_size=$CHUNK \
    --policy.n_action_steps=$CHUNK \
    --policy.push_to_hub=false \
    --steps=6000 \
    --policy.optimizer_lr=$LR \
    --policy.device=cuda \
    --batch_size=32 \
    --log_freq=20 \
    --save_freq=1500 \
    "${@:5}" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'


rsync -ahP $OUTDIR jeremiah@143.215.128.151:/data3/lerobot_checkpoints/
