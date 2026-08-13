#!/bin/bash
#SBATCH --job-name=pi05_iql
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --qos=short
#SBATCH --exclude=ig-88,megazord,cyborg,megazord,sonny,spd-13

echo "Hostname: $(hostname)"
if ! GPU_STATUS=$(nvidia-smi 2>&1) || [[ "$GPU_STATUS" == *ERR* ]]; then
    echo "GPU is not available:" >&2
    echo "$GPU_STATUS" >&2
    exit 1
fi

JOB_NAME=${1:?Pass JOB_NAME as the first argument}
INIT=${INIT:-smolvla}
N_STEP=${N_STEP:-0}
DISCOUNT=${DISCOUNT:-0.99}
TAU=${TAU:-0.005}
LR=${LR:-1e-5}
TEST_DATASET=${TEST_DATASET:-walle_skywalker_testset_annotated}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DROP_PROPRIOCEPTION_INPUT=${DROP_PROPRIOCEPTION_INPUT:-false}
INPUT_DROPOUT_PERCENT=${INPUT_DROPOUT_PERCENT:-50}
SEED=${SEED:-1000}
if ! [[ "$N_STEP" =~ ^[0-9]+$ ]]; then
    echo "N_STEP must be a non-negative integer, got '$N_STEP'" >&2
    exit 1
fi
if ! DISCOUNT_KEY=$(printf "%.10g" "$DISCOUNT" 2>/dev/null); then
    echo "DISCOUNT must be numeric, got '$DISCOUNT'" >&2
    exit 1
fi
if [ "$DISCOUNT_KEY" = "1" ]; then
    DISCOUNT_KEY=1.0
fi
VALUE_KEY=${VALUE_KEY:-annotation_return_gamma_${DISCOUNT_KEY}}
if [ "$N_STEP" -gt 0 ] && [ "$INIT" != "smolvla" ] && [ "$INIT" != "smolvlm256m" ]; then
    echo "Value bootstrapping (N_STEP > 0) requires INIT=smolvla (or smolvlm256m)" >&2
    exit 1
fi
if [ "$SEED" != "1000" ]; then
    JOB_NAME="${JOB_NAME}_seed_${SEED}"
fi
PALIGEMMA_PRETRAINED_PATH=google/paligemma-3b-pt-224
SMOLVLM_PRETRAINED_PATH=HuggingFaceTB/SmolVLM2-256M-Video-Instruct
PI05_BASE_PRETRAINED_PATH=/coc/testnvme/jcoholich3/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30
OUTDIR=./outputs/$JOB_NAME
DATASET=${DATASET:-plug5_offline_rl_dataset_annotated}
DATA_ROOT=/coc/testnvme/jcoholich3/lerobot_data
# DATA_ROOT=/data3/lerobot_data

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
echo "Value key: $VALUE_KEY"
echo "Init: $INIT"
echo "N-step return horizon: $N_STEP"
echo "Discount factor: $DISCOUNT"
echo "Reward key: annotation_reward"
echo "Target network tau: $TAU"
echo "Learning rate: $LR"
echo "Test dataset: $TEST_DATASET"
echo "Weight decay: $WEIGHT_DECAY"
echo "Drop proprioception input: $DROP_PROPRIOCEPTION_INPUT"
echo "Input dropout percent: $INPUT_DROPOUT_PERCENT"
echo "Seed: $SEED"

if [ "$INIT" = "paligemma" ]; then
    INIT_ARGS=(--policy.paligemma_pretrained_path="$PALIGEMMA_PRETRAINED_PATH")
    echo "PaliGemma pretrained path: $PALIGEMMA_PRETRAINED_PATH"
elif [ "$INIT" = "pi05" ]; then
    INIT_ARGS=(--policy.pretrained_path="$PI05_BASE_PRETRAINED_PATH")
    echo "PI05 pretrained path: $PI05_BASE_PRETRAINED_PATH"
elif [ "$INIT" = "smolvla" ] || [ "$INIT" = "smolvlm256m" ]; then
    INIT_ARGS=(
        --policy.value_backbone=smolvlm
        --policy.smolvlm_pretrained_path="$SMOLVLM_PRETRAINED_PATH"
    )
    echo "SmolVLM pretrained path: $SMOLVLM_PRETRAINED_PATH"
else
    echo "Unknown init '$INIT' (expected 'paligemma', 'pi05', or 'smolvla')" >&2
    exit 1
fi

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

export PYTHONPATH="$PWD/src:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=$DATASET \
    --dataset.root="$DATA_ROOT/$DATASET" \
    --test_dataset.repo_id=$TEST_DATASET \
    --test_dataset.root="$DATA_ROOT/$TEST_DATASET" \
    --policy.type=pi05 \
    --seed=$SEED \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    "${INIT_ARGS[@]}" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=value_fn_annotated \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.use_value_model=true \
    --policy.value_key="$VALUE_KEY" \
    --policy.value_dim=1 \
    --policy.value_bootstrap_steps="$N_STEP" \
    --policy.value_discount="$DISCOUNT" \
    --policy.value_reward_key=annotation_reward \
    --policy.value_target_tau="$TAU" \
    --steps=3000 \
    --policy.optimizer_lr=$LR \
    --policy.scheduler_warmup_steps=3000 \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.drop_proprioception_input=$DROP_PROPRIOCEPTION_INPUT \
    --policy.input_dropout_percent=$INPUT_DROPOUT_PERCENT \
    --policy.device=cuda \
    --batch_size=256 \
    --test_batch_size=128 \
    --test_freq=20 \
    --test_first_step=true \
    --test_frame_stride=10 \
    --log_freq=20 \
    --log_first_step=true \
    --save_freq=0 \
    --save_best_test_checkpoint=true \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'

# sbatch --array=1,5,45,50,51,52,53,54,55,56,57,58,59 pi05_value_inference_static.bash "$JOB_NAME" plug5_offline_rl_dataset last 1,5,45,50,51,52,53,54,55,56,57,58,59
