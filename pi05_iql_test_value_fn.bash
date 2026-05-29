#!/bin/bash
#SBATCH --job-name=pi05_iql
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=short

JOB_NAME=$1
VALUE_KEY=${2:-returns_gamma_0.995}
OUTDIR=./outputs/$JOB_NAME
CHUNK=100
LR=5e-5
DATASET='plug5_offline_rl_dataset'
DATA_ROOT=/coc/testnvme/jcoholich3/lerobot_data
# DATA_ROOT=/data3/lerobot_data

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
echo "Value key: $VALUE_KEY"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

export PYTHONPATH="$PWD/src:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=$DATASET \
    --dataset.root="$DATA_ROOT/$DATASET" \
    --policy.type=pi05 \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=lerobot_iql \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.use_value_model=true \
    --policy.value_key="$VALUE_KEY" \
    --policy.value_dim=1 \
    --steps=6000 \
    --policy.optimizer_lr=$LR \
    --policy.device=cuda \
    --batch_size=32 \
    --log_freq=5 \
    --save_freq=1000 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'
