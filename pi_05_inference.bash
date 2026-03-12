rm -rf /home/jeremiah/.cache/huggingface/lerobot/dummy

export CUDA_VISIBLE_DEVICES=1

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --dataset.push_to_hub=false \
  --dataset.root='/home/jeremiah/.cache/huggingface/lerobot/dummy' \
  --dataset.repo_id=dummy/eval_dummy \
  --dataset.single_task="Pick up the dark red block" \
  --dataset.episode_time_s=30000 \
  --dataset.num_episodes=1 \
  --policy.dtype=bfloat16 \
  --policy.path=/home/jeremiah/lerobot/outputs/eve_blocks_6x_longer/checkpoints/003000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/chunk_100/checkpoints/003000/pretrained_model
  # --policy.type=pi05 \