#!/usr/bin/env bash
set -e

# Plug the charger into the power strip
# Unplug the charger
# Unscrew the nut and set it on the table
# Thread the nut onto the bolt

RECORD_NAME="${1:-last_recording}"

POLICY_SERVER_ADDRESS="${POLICY_SERVER_ADDRESS:-127.0.0.1:8080}"
POLICY_PATH="${POLICY_PATH:-/home/jeremiah/lerobot/outputs/unplug3_bc_and_dagger/checkpoints/003000/pretrained_model}"
TASK="${TASK:-Unplug the charger}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
INFERENCE_FPS="${INFERENCE_FPS:-30}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-100}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-weighted_average}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-100}"

export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH:-}"

python -m lerobot.async_inference.robot_client \
  --server_address="${POLICY_SERVER_ADDRESS}" \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --robot.record="${RECORD_NAME}" \
  --task="${TASK}" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${POLICY_PATH}" \
  --policy_device="${POLICY_DEVICE}" \
  --client_device="${CLIENT_DEVICE}" \
  --fps="${INFERENCE_FPS}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --policy_dtype="${POLICY_DTYPE}" \
  --policy_n_action_steps="${POLICY_N_ACTION_STEPS}"

# Alternate checkpoints:
# POLICY_PATH=/home/jeremiah/lerobot/outputs/plug3_unplug3_bc_and_dagger_12k_4GPU_double_lr/checkpoints/006000/pretrained_model bash pi_05_inference.bash
# POLICY_PATH=/home/jeremiah/lerobot/outputs/thread3_unthread4_bc_and_dagger_12k_4GPU_double_lr/checkpoints/003000/pretrained_model bash pi_05_inference.bash
