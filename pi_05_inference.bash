# Usage: bash pi_05_inference.bash [record_name] [none|base|PIVOT|pivot|primitive|ensemble|eve] [parallel|serial]
RECORD_NAME="${1:-last_recording}"
INTERVENTIONS="${2:-none}"
ENSEMBLE_REQUEST_MODE="${3:-parallel}"
if [[ "${ENSEMBLE_REQUEST_MODE}" == "sequential" ]]; then
  ENSEMBLE_REQUEST_MODE="serial"
fi
if [[ "${ENSEMBLE_REQUEST_MODE}" != "parallel" && "${ENSEMBLE_REQUEST_MODE}" != "serial" ]]; then
  echo "Ensemble request mode must be parallel or serial" >&2
  exit 2
fi
if [[ "${INTERVENTIONS}" == "eve" ]]; then
  INTERVENTIONS="ensemble"
elif [[ "${INTERVENTIONS}" == "base" ]]; then
  INTERVENTIONS="none"
elif [[ "${INTERVENTIONS}" == "pivot" ]]; then
  INTERVENTIONS="PIVOT"
fi
export LEROBOT_DEMO_NAME="${RECORD_NAME}"

export PYTHONPATH="/home/jeremiah/openteach:${PYTHONPATH}"

# MMD gamma: median (adaptive), max_eig, or a positive fixed number.
# --interventions options (case-sensitive): none, PIVOT, primitive, ensemble.
# ssh -N -f -L localhost:46049:localhost:46049 -J jcoholich3@sky1.cc.gatech.edu jcoholich3@heistotron.cc.gatech.edu
# ssh -N -f -L localhost:51995:localhost:51995 -J jcoholich3@sky1.cc.gatech.edu jcoholich3@optimistprime.cc.gatech.edu
python src/lerobot/scripts/pi05_inference.py \
  --policy_server="${LEROBOT_POLICY_SERVER:-/tmp/lerobot-pi05-${UID}.sock}" \
  --robot.type=franka \
  --robot.id=franka \
  --robot.port=dummy \
  --robot.record="${RECORD_NAME}" \
  --task='place the coffee pod in the bin' \
  --duration=30000 \
  --interventions="${INTERVENTIONS}" \
  --ensemble_request_mode="${ENSEMBLE_REQUEST_MODE}" \
  --vlm_server_url="https://api.openai.com" \
  --vlm_model_name="gpt-6-astra" \
  --manual_guidance=false \
  --vis_spreads=false \
  --policy.dtype=bfloat16 \
  --policy.n_action_steps=75 \
  --policy.mmd_gamma="${LEROBOT_MMD_GAMMA:-median}" \
  --policy.diversity_gamma="${LEROBOT_DIVERSITY_GAMMA:-0.06302211495246096}" \
  --policy.path=/home/jeremiah/lerobot/outputs/both_in_bin_interleaved/checkpoints/003000/pretrained_model
  # --policy.path=/home/jeremiah/lerobot/outputs/both_in_bin_interleaved/checkpoints/003000/pretrained_model
  # --policy.type=pi05 \

# prompts: place both blocks in the bin
# place the pink block, then the blue block in the bin
# place the blue block, then the pink block in the bin

# --vlm_model_name="Qwen/Qwen3.8-27B" \

# --vlm_server_url="http://127.0.0.1:46049" \
# --vlm_model_name="Qwen/Qwen3-VL-8B-Instruct" \
