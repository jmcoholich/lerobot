rm -rf /home/jeremiah/.cache/huggingface/lerobot/dummy

RECORD_NAME="${1:-last_recording}"
TASK="${2:-place the coffee pod in the bin}"
LEROBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${LEROBOT_ROOT}/src:/home/jeremiah/openteach${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOMONKEY_SERVER="${ROBOMONKEY_SERVER:-http://127.0.0.1:8991}"
export ROBOMONKEY_TIMEOUT_S="${ROBOMONKEY_TIMEOUT_S:-60}"
export ROBOMONKEY_TIMING_ROOT="${ROBOMONKEY_TIMING_ROOT:-/home/jeremiah/openteach/extracted_data}"

python "${LEROBOT_ROOT}/src/lerobot/scripts/lerobot_record.py" \
  --robot.record="${RECORD_NAME}" \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --dataset.push_to_hub=false \
  --dataset.root='/home/jeremiah/.cache/huggingface/lerobot/dummy' \
  --dataset.repo_id=dummy/eval_dummy \
  --dataset.single_task="place the coffee pod in the bin" \
  --dataset.episode_time_s=30000 \
  --dataset.num_episodes=1 \
  --policy.dtype=bfloat16 \
  --policy.path=/home/jeremiah/lerobot/outputs/both_in_bin_interleaved/checkpoints/003000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/both_in_bin_interleaved/checkpoints/003000/pretrained_model
  # --policy.type=pi05 \

# prompts: place both blocks in the bin
# place the pink block, then the blue block in the bin
# place the blue block, then the pink block in the bin
# "place the coffee pod in the bin"
# "place the travel adapter in the bin"
