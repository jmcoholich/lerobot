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
  --dataset.single_task="Unplug the charger" \
  --dataset.episode_time_s=60000 \
  --dataset.num_episodes=1 \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=100 \
  --policy.path=/home/jeremiah/lerobot/outputs/plug3/checkpoints/006000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/plug3_unplug3_bc_and_dagger_12k_4GPU_double_lr/checkpoints/006000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/thread3_unthread4_bc_and_dagger_12k_4GPU_double_lr/checkpoints/003000/pretrained_model
  # --dataset.single_task="Plug the charger into the power strip" \
  # --policy.type=pi05 \
