#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <record_name>"
    echo "Example: $0 plug_fwd_11"
    exit 1
fi

set -m

RECORD_NAME="$1"
DATA_PID=""
INFERENCE_PID=""

stop_children() {
    trap - INT

    echo
    echo "Stopping pi_05_inference.bash..."
    kill -INT -- "-${INFERENCE_PID}" 2>/dev/null || true

    echo "Stopping data_collect.py..."
    kill -INT -- "-${DATA_PID}" 2>/dev/null || true

    wait "${INFERENCE_PID}" 2>/dev/null || true
    wait "${DATA_PID}" 2>/dev/null || true
    exit 130
}

trap stop_children INT

echo "Starting data_collect.py..."
bash -c 'exec python "$@"' bash /home/jeremiah/openteach/data_collect.py robot=franka demo_num="${RECORD_NAME}" &
DATA_PID=$!

sleep 0.2

echo "Starting pi_05_inference.bash..."
bash pi_05_inference.bash "${RECORD_NAME}" &
INFERENCE_PID=$!

echo "Started:"
echo "  data_collect.py PID: ${DATA_PID}"
echo "  inference      PID: ${INFERENCE_PID}"
echo
echo "Press Ctrl+C to stop pi_05_inference.bash, then data_collect.py."

wait
