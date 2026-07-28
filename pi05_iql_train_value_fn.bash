#!/bin/bash
#SBATCH --job-name=pi05_iql
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --exclude=ig-88,megazord,cyborg

if ! GPU_STATUS=$(nvidia-smi 2>&1) || [[ "$GPU_STATUS" == *ERR* ]]; then
    echo "GPU is not available:" >&2
    echo "$GPU_STATUS" >&2
    exit 1
fi

JOB_NAME=${1:?Pass JOB_NAME as the first argument}
VALUE_KEY=${VALUE_KEY:-sparse_returns_gamma_1.0}
INIT=${INIT:-paligemma}
TEST_DATASET=${TEST_DATASET:-walle_skywalker_testset}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DROP_PROPRIOCEPTION_INPUT=${DROP_PROPRIOCEPTION_INPUT:-false}
INPUT_DROPOUT_PERCENT=${INPUT_DROPOUT_PERCENT:-0}
SEED=${SEED:-1000}
if [ "$SEED" != "1000" ]; then
    JOB_NAME="${JOB_NAME}_seed_${SEED}"
fi
PALIGEMMA_PRETRAINED_PATH=google/paligemma-3b-pt-224
SMOLVLM_PRETRAINED_PATH=HuggingFaceTB/SmolVLM2-256M-Video-Instruct
PI05_BASE_PRETRAINED_PATH=/coc/testnvme/jcoholich3/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30
OUTDIR=./outputs/$JOB_NAME
LR=1e-5
DATASET='plug5_offline_rl_dataset'
DATA_ROOT=/coc/testnvme/jcoholich3/lerobot_data
# DATA_ROOT=/data3/lerobot_data

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
echo "Value key: $VALUE_KEY"
echo "Init: $INIT"
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
    --wandb.project=lerobot_iql \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.use_value_model=true \
    --policy.value_key="$VALUE_KEY" \
    --policy.value_dim=1 \
    --steps=3000 \
    --policy.optimizer_lr=$LR \
    --policy.scheduler_warmup_steps=3000 \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.drop_proprioception_input=$DROP_PROPRIOCEPTION_INPUT \
    --policy.input_dropout_percent=$INPUT_DROPOUT_PERCENT \
    --policy.device=cuda \
    --batch_size=32 \
    --test_freq=100 \
    --test_first_step=true \
    --test_frame_stride=10 \
    --log_freq=100 \
    --log_first_step=true \
    --save_freq=0 \
    --save_best_test_checkpoint=true \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'

# sbatch --array=1,5,45,50,51,52,53,54,55,56,57,58,59 pi05_value_inference_static.bash "$JOB_NAME" plug5_offline_rl_dataset last 1,5,45,50,51,52,53,54,55,56,57,58,59
