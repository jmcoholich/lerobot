#!/usr/bin/env bash
set -ex

N="${1:-1}"
PREFIX="grogu_6k"
CHECKPOINT=6000

wait_for_key() {
  read -r -n 1 -p "Press any key to continue..." </dev/tty
  echo
}

while true; do
  python /home/jeremiah/deoxys_control/deoxys/examples/reset_robot_joints.py --side --eval >/dev/null 2>&1
  python /home/jeremiah/openteach/reset_gripper.py
  # wait_for_key

  bash pi_05_server_inference.bash \
    --prompt "Stack the four blocks" \
    --name stackFour \
    --chunk-size 40 \
    --checkpoint ${CHECKPOINT} \
    --port 8080 \
    --record "${PREFIX}_stackFour_${N}"
  wait_for_key

  python /home/jeremiah/deoxys_control/deoxys/examples/reset_robot_joints.py --side --eval >/dev/null 2>&1
  python /home/jeremiah/openteach/reset_gripper.py
  # wait_for_key

  bash pi_05_server_inference.bash \
    --prompt "Unstack the four blocks" \
    --name unstackFour \
    --chunk-size 40 \
    --checkpoint ${CHECKPOINT} \
    --port 8081 \
    --record "${PREFIX}_unstackFour_${N}"
  wait_for_key

  N=$((N + 1))
done
