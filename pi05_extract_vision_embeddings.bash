#!/bin/bash
#SBATCH --job-name=pi05_extract_vision_embeddings
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=long

DATASET='eve_blocks_6x_abs_joint'

echo "Dataset: $DATASET"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

python src/lerobot/scripts/lerobot_extract_vision_embeddings.py \
    --dataset.repo_id=$DATASET \
    --dataset.root="/coc/testnvme/jcoholich3/lerobot_data/$DATASET" \
    --policy.type=pi05 \
    --policy.pretrained_path=jcoholich/pi05_droid_converted \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --batch_size=32
