rm -rf /home/jeremiah/.cache/huggingface/lerobot/dummy

RECORD_NAME="${1:-last_recording}"

export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --robot.record="${RECORD_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.root='/home/jeremiah/.cache/huggingface/lerobot/dummy' \
  --dataset.repo_id=dummy/eval_dummy \
  --dataset.episode_time_s=60000 \
  --dataset.num_episodes=1 \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=50 \
  --dataset.single_task="Unplug the charger" \
  --policy.path=/home/jeremiah/lerobot/outputs/unplug3/checkpoints/003000/pretrained_model
  # --dataset.single_task="Unscrew the nut and set it on the table" \
  # --policy.path=/home/jeremiah/lerobot/outputs/unthread4/checkpoints/006000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/unthread4/checkpoints/004000/pretrained_model
  # --dataset.single_task="Unscrew the nut and set it on the table" \
  # --policy.type=pi05 \
