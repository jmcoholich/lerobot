#!/usr/bin/env bash
set -e
set -m
export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH}"
usage() {
  cat <<'EOF'
Usage: bash pi_05_server_inference.bash --prompt TEXT --name POLICY_NAME [options]

Options:
  --port PORT           Policy server port on localhost (default: 8080)
  --prompt TEXT         Task prompt to pass as --task (required)
  --record NAME         Recording name (default: last_recording)
  --name POLICY_NAME    Policy output directory name under outputs/ (required)
  --checkpoint STEP     Checkpoint step number (default: 3000)
  --n-action-steps N    Policy n_action_steps override (default: 100)
  --help                Show this help message
EOF
}

PORT="8080"
PROMPT=""
RECORD="last_recording"
NAME=""
CHECKPOINT="3000"
N_ACTION_STEPS="100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --record)
      RECORD="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --n-action-steps)
      N_ACTION_STEPS="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PROMPT}" ]]; then
  echo "Missing required argument: --prompt" >&2
  usage >&2
  exit 1
fi

if [[ -z "${NAME}" ]]; then
  echo "Missing required argument: --name" >&2
  usage >&2
  exit 1
fi

if [[ ! "${CHECKPOINT}" =~ ^[0-9]+$ ]]; then
  echo "--checkpoint must be a step number, got: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! "${N_ACTION_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "--n-action-steps must be a number, got: ${N_ACTION_STEPS}" >&2
  exit 1
fi

CHECKPOINT_DIR="$(printf "%06d" "${CHECKPOINT}")"
POLICY_PATH="/home/jeremiah/lerobot/outputs/${NAME}/checkpoints/${CHECKPOINT_DIR}/pretrained_model"

DATA_PID=""
ROBOT_CLIENT_PID=""

stop_children() {
  trap - INT

  echo
  echo "Stopping robot_client..."
  [[ -n "${ROBOT_CLIENT_PID}" ]] && kill -INT -- "-${ROBOT_CLIENT_PID}" 2>/dev/null || true

  echo "Stopping data_collect.py..."
  [[ -n "${DATA_PID}" ]] && kill -INT -- "-${DATA_PID}" 2>/dev/null || true

  [[ -n "${ROBOT_CLIENT_PID}" ]] && wait "${ROBOT_CLIENT_PID}" 2>/dev/null || true
  [[ -n "${DATA_PID}" ]] && wait "${DATA_PID}" 2>/dev/null || true
  exit 130
}

trap stop_children INT

echo "Starting data_collect.py..."
bash -c 'exec python "$@"' bash /home/jeremiah/openteach/data_collect.py robot=franka demo_num="${RECORD}" &
DATA_PID=$!

sleep 0.2

echo "Starting robot_client..."
python -m lerobot.async_inference.robot_client \
  --server_address="localhost:${PORT}" \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --robot.record="${RECORD}" \
  --task="${PROMPT}" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${POLICY_PATH}" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=100 \
  --chunk_size_threshold=0.95 \
  --aggregate_fn_name=weighted_average \
  --policy_dtype=bfloat16 \
  --policy_n_action_steps="${N_ACTION_STEPS}" &
ROBOT_CLIENT_PID=$!

echo "Started:"
echo "  data_collect.py PID: ${DATA_PID}"
echo "  robot_client    PID: ${ROBOT_CLIENT_PID}"

set +e
wait "${ROBOT_CLIENT_PID}"
ROBOT_CLIENT_STATUS=$?
set -e

echo "Stopping data_collect.py..."
kill -INT -- "-${DATA_PID}" 2>/dev/null || true
wait "${DATA_PID}" 2>/dev/null || true

exit "${ROBOT_CLIENT_STATUS}"
