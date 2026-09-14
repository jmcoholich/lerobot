#!/usr/bin/env bash

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <record_name> [none|base|PIVOT|pivot|primitive|ensemble|eve] [parallel|serial]"
    echo "Example: $0 test_eve_7 eve serial"
    exit 1
fi

set -m

RECORD_NAME="$1"
INTERVENTIONS="${2:-none}"
INFERENCE_ARGS=("${RECORD_NAME}" "${INTERVENTIONS}")
if [[ $# -eq 3 ]]; then
    ENSEMBLE_REQUEST_MODE="$3"
    if [[ "${ENSEMBLE_REQUEST_MODE}" == "sequential" ]]; then
        ENSEMBLE_REQUEST_MODE="serial"
    fi
    if [[ "${ENSEMBLE_REQUEST_MODE}" != "parallel" && "${ENSEMBLE_REQUEST_MODE}" != "serial" ]]; then
        echo "Ensemble request mode must be parallel or serial" >&2
        exit 2
    fi
    INFERENCE_ARGS+=("${ENSEMBLE_REQUEST_MODE}")
fi
OPENTEACH_DATA_DIR="/home/jeremiah/openteach/extracted_data"
export LEROBOT_TRAJECTORY_DIR="${LEROBOT_TRAJECTORY_DIR:-${OPENTEACH_DATA_DIR}/demonstration_${RECORD_NAME}}"
DATA_PID=""
INFERENCE_PID=""

stop_children() {
    # Let both process groups finish saving without interrupting cleanup again.
    trap '' INT TERM

    echo
    if [[ -n "${INFERENCE_PID}" ]]; then
        echo "Stopping pi_05_inference.bash..."
        kill -INT -- "-${INFERENCE_PID}" 2>/dev/null || true
    fi

    if [[ -n "${DATA_PID}" ]]; then
        echo "Stopping data_collect.py..."
        kill -INT -- "-${DATA_PID}" 2>/dev/null || true
    fi

    if [[ -n "${INFERENCE_PID}" ]]; then
        wait "${INFERENCE_PID}" 2>/dev/null || true
    fi
    if [[ -n "${DATA_PID}" ]]; then
        wait "${DATA_PID}" 2>/dev/null || true
    fi
}

trap stop_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting data_collect.py..."
bash -c 'exec python "$@"' bash /home/jeremiah/openteach/data_collect.py robot=franka demo_num="${RECORD_NAME}" storage_path="${OPENTEACH_DATA_DIR}" &
DATA_PID=$!

sleep 0.2

echo "Starting pi_05_inference.bash..."
bash pi_05_inference.bash "${INFERENCE_ARGS[@]}" &
INFERENCE_PID=$!

echo "Started:"
echo "  data_collect.py PID: ${DATA_PID}"
echo "  inference      PID: ${INFERENCE_PID}"
echo "  trajectory logs: ${LEROBOT_TRAJECTORY_DIR}"
echo
echo "Press Ctrl+C to stop pi_05_inference.bash, then data_collect.py."

# Stop the other process when either inference or data collection ends.
wait -n "${INFERENCE_PID}" "${DATA_PID}"
ROLLOUT_STATUS=$?
echo "A rollout process exited (status ${ROLLOUT_STATUS}); stopping the remaining processes."
exit "${ROLLOUT_STATUS}"
