#!/bin/bash
#SBATCH --job-name=pi05_value_infer
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --array=0-297%20

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

export PYTHONPATH="$PWD/src:${PYTHONPATH}"

CHECKPOINT=${3:-last}
RUN_NAME="${1}_${CHECKPOINT}_${2}"
VIDEO_PREFIX="output_videos/${2}_all_cams"
VIDEO_PATH="/coc/testnvme/jcoholich3/reward-modeling/${VIDEO_PREFIX}/task_0/episode_${SLURM_ARRAY_TASK_ID}/all_cams.mp4"
VIDEO_ARGS=()
if [ -f "$VIDEO_PATH" ]; then
  VIDEO_ARGS+=(--skip-video)
fi

python src/lerobot/scripts/lerobot_pi05_value_inference.py \
  --policy-path=outputs/$1/checkpoints/$CHECKPOINT/pretrained_model \
  --dataset-root=/coc/testnvme/jcoholich3/lerobot_data/$2 \
  --repo-id=$2 \
  --output-dir=/coc/testnvme/jcoholich3/reward-modeling/viewer_files \
  --manifest-name="${RUN_NAME}.json" \
  --video-prefix="$VIDEO_PREFIX" \
  --episodes="$SLURM_ARRAY_TASK_ID" \
  --batch-size=48 \
  --skip-manifest \
  "${VIDEO_ARGS[@]}"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  python - "$RUN_NAME" <<'PY'
import json
import sys

run_name = sys.argv[1]
path = f"/coc/testnvme/jcoholich3/reward-modeling/viewer_files/{run_name}.json"
entries = [f"{run_name}/task_0/episode_{i}/all_cams" for i in range(298)]
with open(path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)
PY
fi
