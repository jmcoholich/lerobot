#!/bin/bash
#SBATCH --job-name=pi05_value_infer
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G 2080_ti:1
#SBATCH -c 6
#SBATCH --mem=12G
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --array=0-297%20
#SBATCH --exclude=ig-88,megazord,cyborg,megazord,sonny,spd-13,flexo,friday,johnny5

set -e

source /coc/testnvme/$USER/.bashrc
conda activate lerobot
cd /coc/testnvme/$USER/lerobot_iql
export PYTHONPATH="$PWD/src:${PYTHONPATH}"

# Keep Hugging Face dataset locks off the shared filesystem and separate per task.
HF_DATASETS_CACHE=$(mktemp -d /tmp/lerobot-inference-cache.XXXXXX)
export HF_DATASETS_CACHE
trap 'rm -rf -- "$HF_DATASETS_CACHE"' EXIT

echo "Hostname: $(hostname)"
if ! GPU_STATUS=$(nvidia-smi 2>&1) || [[ "$GPU_STATUS" == *ERR* ]]; then
    echo "GPU is not available:" >&2
    echo "$GPU_STATUS" >&2
    exit 1
fi

CHECKPOINT="${3:-last}"
RUN_NAME="${INFERENCE_RUN_NAME:-$2}"
EPISODES="${4:-$SLURM_ARRAY_TASK_ID}"
VIDEO_PREFIX="output_videos/${2}_all_cams"
VIDEO_ARGS=()
if [ "${SKIP_VIDEO:-0}" = 1 ]; then
  VIDEO_ARGS+=(--skip-video)
fi

python src/lerobot/scripts/lerobot_pi05_value_inference.py \
  --policy-path="${POLICY_PATH:-outputs/$1/checkpoints/$CHECKPOINT/pretrained_model}" \
  --dataset-root="${DATASET_ROOT:-/coc/testnvme/jcoholich3/lerobot_data/$2}" \
  --output-dir="${INFERENCE_OUTPUT_DIR:-/coc/testnvme/jcoholich3/reward-modeling/viewer_files}" \
  --manifest-name="${RUN_NAME}.json" \
  --video-prefix="$VIDEO_PREFIX" \
  --episodes="$SLURM_ARRAY_TASK_ID" \
  --batch-size=48 \
  --skip-manifest \
  "${VIDEO_ARGS[@]}"

if [ "${SKIP_MANIFEST:-0}" != 1 ] && [ "$SLURM_ARRAY_TASK_ID" = "$SLURM_ARRAY_TASK_MIN" ]; then
  python - "$RUN_NAME" "$EPISODES" "${INFERENCE_OUTPUT_DIR:-/coc/testnvme/jcoholich3/reward-modeling/viewer_files}" <<'PY'
import sys
from pathlib import Path

from lerobot.scripts.lerobot_pi05_value_inference import write_json_atomic

run_name = sys.argv[1]
episodes = [int(ep) for ep in sys.argv[2].split(",")]
path = Path(sys.argv[3]) / f"{run_name}.json"
entries = [f"{run_name}/task_0/episode_{i}/all_cams" for i in episodes]
write_json_atomic(path, entries)
PY
fi
