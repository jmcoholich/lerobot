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
cd /coc/testnvme/$USER/lerobot_iql
export PYTHONPATH="$PWD/src:${PYTHONPATH}"

CHECKPOINT=${3:-last}
RUN_NAME="$2"
EPISODES=${4:-$SLURM_ARRAY_TASK_ID}
VIDEO_PREFIX="output_videos/${2}_all_cams"
VIDEO_PATH="/coc/testnvme/jcoholich3/reward-modeling/${VIDEO_PREFIX}/task_0/episode_${SLURM_ARRAY_TASK_ID}/all_cams.mp4"
VIDEO_ARGS=()
if [ -f "$VIDEO_PATH" ]; then
  VIDEO_ARGS+=(--skip-video)
fi

python src/lerobot/scripts/lerobot_pi05_value_inference.py \
  --policy-path=outputs/$1/checkpoints/$CHECKPOINT/pretrained_model \
  --dataset-root=/coc/testnvme/jcoholich3/lerobot_data/$2 \
  --output-dir=/coc/testnvme/jcoholich3/reward-modeling/viewer_files \
  --manifest-name="${RUN_NAME}.json" \
  --video-prefix="$VIDEO_PREFIX" \
  --episodes="$SLURM_ARRAY_TASK_ID" \
  --batch-size=48 \
  --skip-manifest \
  "${VIDEO_ARGS[@]}"

if [ "$SLURM_ARRAY_TASK_ID" = "$SLURM_ARRAY_TASK_MIN" ]; then
  python - "$RUN_NAME" "$EPISODES" <<'PY'
import json
import sys

run_name = sys.argv[1]
episodes = [int(ep) for ep in sys.argv[2].split(",")]
path = f"/coc/testnvme/jcoholich3/reward-modeling/viewer_files/{run_name}.json"
entries = [f"{run_name}/task_0/episode_{i}/all_cams" for i in episodes]
with open(path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)
PY
fi
