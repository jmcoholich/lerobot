#!/bin/bash
#SBATCH --job-name=awr_merge
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --kill-on-invalid-dep=yes

set -e

PLAN="${1:?Pass the inference plan.json path as the first argument}"
source "/coc/testnvme/$USER/.bashrc"
conda activate lerobot
cd "/coc/testnvme/$USER/lerobot_iql"

python -u src/lerobot/scripts/lerobot_pi05_value_awr.py merge "$PLAN"
