#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH}"
exec python -m lerobot.scripts.pi05_policy_server \
  --socket="${LEROBOT_POLICY_SERVER:-/tmp/lerobot-pi05-${UID}.sock}" "$@"
