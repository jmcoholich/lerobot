#!/bin/bash
#SBATCH --job-name=pi05_training
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=long

JOB_NAME=$1
OUTDIR=./outputs/$JOB_NAME
CHUNK=100
LR=5e-5

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=bimodal_blocks_in_bin \
    --dataset.root='/coc/testnvme/jcoholich3/lerobot_data/bimodal_blocks_in_bin' \
    --policy.type=pi05 \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.chunk_size=$CHUNK \
    --policy.n_action_steps=$CHUNK \
    --steps=6000 \
    --policy.optimizer_lr=$LR \
    --policy.device=cuda \
    --batch_size=32 \
    --log_freq=5 \
    --save_freq=500 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'


