rm -rf /home/jeremiah/.cache/huggingface/lerobot/dummy

RECORD_NAME="${1:-last_recording}"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --robot.record="${RECORD_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.root='/home/jeremiah/.cache/huggingface/lerobot/dummy' \
  --dataset.repo_id=dummy/eval_dummy \
  --dataset.single_task="Unplug the charger" \
  --dataset.episode_time_s=60000 \
  --dataset.num_episodes=1 \
  --policy.dtype=bfloat16 \
  --policy.path=/home/jeremiah/lerobot/outputs/plug_both_reversed/checkpoints/003000/pretrained_model
  # --dataset.single_task="Plug the charger into the power strip" \
  # --policy.n_action_steps=20
  # --policy.type=pi05 \
