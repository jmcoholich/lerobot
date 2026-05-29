#!/bin/bash
#SBATCH --job-name=pi05_value_infer
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --array=0-297

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

export PYTHONPATH="$PWD/src:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_pi05_value_inference.py \
  --policy-path=outputs/plug5_value_fn/checkpoints/005000/pretrained_model \
  --dataset-root=/coc/testnvme/jcoholich3/lerobot_data/plug5_offline_rl_dataset \
  --repo-id=plug5_offline_rl_dataset \
  --output-dir=/coc/testnvme/jcoholich3/reward-modeling/viewer_files \
  --manifest-name=plug5_value_fn.json \
  --video-prefix=output_videos/plug5_value_fn \
  --episodes="$SLURM_ARRAY_TASK_ID" \
  --batch-size=48 \
  --skip-manifest

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  python - <<'PY'
import json
path = "/coc/testnvme/jcoholich3/reward-modeling/viewer_files/plug5_value_fn.json"
entries = [f"plug5_value_fn/task_0/episode_{i}/all_cams" for i in range(298)]
with open(path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)
PY
fi
