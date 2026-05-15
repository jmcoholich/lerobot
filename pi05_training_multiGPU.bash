#!/bin/bash
#SBATCH --job-name=pi05_training
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:4
#SBATCH --cpus-per-gpu=10
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=16G
#SBATCH -x nestor,chappie,ig-88,perseverance

JOB_NAME=$2
OUTDIR=./outputs/$JOB_NAME
CHUNK=100
DATASET=$1

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
LR=1e-4

nvidia-smi

echo "Using $NUM_GPUS GPUs with a learning rate of $LR"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot
accelerate launch \
--multi_gpu \
--num_processes=$NUM_GPUS \
--mixed_precision=bf16 \
$(which lerobot-train) \
    --dataset.repo_id=$DATASET \
    --dataset.root="/coc/testnvme/jcoholich3/lerobot_data/$DATASET" \
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
    --policy.optimizer_lr=$LR \
    --steps=6000 \
    --policy.device=cuda \
    --batch_size=24 \
    --log_freq=5 \
    --save_freq=1500 \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'

rsync -ahP $OUTDIR jeremiah@143.215.128.151:/data3/lerobot_checkpoints/
