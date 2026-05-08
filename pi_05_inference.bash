rm -rf /home/jeremiah/.cache/huggingface/lerobot/dummy

export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --dataset.push_to_hub=false \
  --dataset.root='/home/jeremiah/.cache/huggingface/lerobot/dummy' \
  --dataset.repo_id=dummy/eval_dummy \
  --dataset.single_task="place both blocks in the bin" \
  --dataset.episode_time_s=30000 \
  --dataset.num_episodes=1 \
  --policy.dtype=bfloat16 \
  --policy.path=/home/jeremiah/lerobot/outputs/both_in_bin_interleaved/checkpoints/003000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/both_in_bin_interleaved/checkpoints/003000/pretrained_model
  # --policy.type=pi05 \

# prompts: place both blocks in the bin
# place the pink block, then the blue block in the bin
# place the blue block, then the pink block in the bin
