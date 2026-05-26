#!/usr/bin/env bash
set -ex

N="${1:-1}"
PREFIX="rollout"

wait_for_key() {
  read -r -n 1 -p "Press any key to continue..." </dev/tty
  echo
}

while true; do
  python /home/jeremiah/deoxys_control/deoxys/examples/reset_robot_joints.py --side --eval >/dev/null 2>&1
  python /home/jeremiah/openteach/reset_gripper.py
  wait_for_key

  bash pi_05_server_inference.bash \
    --prompt "Plug the charger into the power strip" \
    --name plug5 \
    --chunk-size 40 \
    --checkpoint 3000 \
    --port 8080 \
    --record "${PREFIX}_plug5_${N}"
  wait_for_key

  python /home/jeremiah/deoxys_control/deoxys/examples/reset_robot_joints.py --unplug --eval >/dev/null 2>&1
  python /home/jeremiah/openteach/reset_gripper.py
  wait_for_key

  bash pi_05_server_inference.bash \
    --prompt "Unplug the charger" \
    --name unplug5 \
    --chunk-size 40 \
    --checkpoint 3000 \
    --port 8081 \
    --record "${PREFIX}_unplug5_${N}"
  wait_for_key

  N=$((N + 1))
done
