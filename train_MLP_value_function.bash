#!/bin/bash
#SBATCH --job-name=pi05_vision_mlp
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
INIT=${INIT:-pi05}
N_STEP=${N_STEP:-0}
DISCOUNT=${DISCOUNT:-0.95}
TAU=${TAU:-0.05}
LR=${LR:-3e-5}
TRAIN_EPISODES=${TRAIN_EPISODES:?Set TRAIN_EPISODES to indices (0,2,4) or an end-exclusive range (0:80)}
TEST_EPISODES=${TEST_EPISODES:?Set TEST_EPISODES to indices (1,3,5) or an end-exclusive range (80:100)}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DROP_PROPRIOCEPTION_INPUT=${DROP_PROPRIOCEPTION_INPUT:-false}
BLACKOUT_FRONT_CAMERA_INPUT=${BLACKOUT_FRONT_CAMERA_INPUT:-false}
INPUT_DROPOUT_PERCENT=${INPUT_DROPOUT_PERCENT:-25}
VISION_PROJECTION_DIM=${VISION_PROJECTION_DIM:-256}
MLP_HIDDEN_DIM=${MLP_HIDDEN_DIM:-512}
MLP_DROPOUT=${MLP_DROPOUT:-0.0}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}
STEPS=${STEPS:-7500}
WARMUP_STEPS=${WARMUP_STEPS:-3000}
WANDB_PROJECT=${WANDB_PROJECT:-value_fn_annotated}
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
if [ "$SEED" != "1000" ]; then
    JOB_NAME="${JOB_NAME}_seed_${SEED}"
fi
PALIGEMMA_PRETRAINED_PATH=google/paligemma-3b-pt-224
PI05_BASE_PRETRAINED_PATH=/coc/testnvme/jcoholich3/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30
OUTDIR=./outputs/$JOB_NAME
DATASET=${DATASET:-plug5_offline_rl_dataset_annotated}
DATA_ROOT=/coc/testnvme/jcoholich3/lerobot_data

episode_indices_json() {
    local spec=$1 start end result index
    if [[ "$spec" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[$spec]"
    elif [[ "$spec" =~ ^([0-9]+):([0-9]+)$ ]]; then
        start=${BASH_REMATCH[1]}
        end=${BASH_REMATCH[2]}
        if (( start >= end )); then
            echo "Episode range must be nonempty and end-exclusive, got '$spec'" >&2
            return 1
        fi
        result=$start
        for ((index = start + 1; index < end; index++)); do
            result+=",$index"
        done
        echo "[$result]"
    else
        echo "Episode indices must be comma-separated integers or START:END, got '$spec'" >&2
        return 1
    fi
}

TRAIN_EPISODES_JSON=$(episode_indices_json "$TRAIN_EPISODES") || exit 1
TEST_EPISODES_JSON=$(episode_indices_json "$TEST_EPISODES") || exit 1

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
echo "Value key: $VALUE_KEY"
echo "Vision initialization: $INIT"
echo "N-step return horizon: $N_STEP"
echo "Discount factor: $DISCOUNT"
echo "Reward key: annotation_reward"
echo "Target network tau: $TAU"
echo "Learning rate: $LR"
echo "Train episodes: $TRAIN_EPISODES"
echo "Test episodes: $TEST_EPISODES"
echo "Weight decay: $WEIGHT_DECAY"
echo "Drop proprioception input: $DROP_PROPRIOCEPTION_INPUT"
echo "Blackout front camera input: $BLACKOUT_FRONT_CAMERA_INPUT"
echo "Input dropout percent: $INPUT_DROPOUT_PERCENT"
echo "Vision projection dim: $VISION_PROJECTION_DIM"
echo "MLP hidden dim: $MLP_HIDDEN_DIM"
echo "MLP dropout: $MLP_DROPOUT"
echo "Freeze vision encoder: $FREEZE_VISION_ENCODER"
echo "Training steps: $STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "W&B project: $WANDB_PROJECT"
echo "Seed: $SEED"

if [ "$INIT" = "paligemma" ]; then
    VISION_PRETRAINED_PATH=$PALIGEMMA_PRETRAINED_PATH
elif [ "$INIT" = "pi05" ]; then
    VISION_PRETRAINED_PATH=$PI05_BASE_PRETRAINED_PATH
else
    echo "Unknown init '$INIT' (expected 'paligemma' or 'pi05')" >&2
    exit 1
fi
echo "Vision encoder pretrained path: $VISION_PRETRAINED_PATH"

source /coc/testnvme/$USER/.bashrc
conda activate lerobot

export PYTHONPATH="$PWD/src:${PYTHONPATH}"

python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=$DATASET \
    --dataset.root="$DATA_ROOT/$DATASET" \
    --train_episodes="$TRAIN_EPISODES_JSON" \
    --test_episodes="$TEST_EPISODES_JSON" \
    --policy.type=pi05 \
    --seed=$SEED \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=$FREEZE_VISION_ENCODER \
    --policy.train_expert_only=false \
    --policy.use_value_model=true \
    --policy.value_backbone=vision_mlp \
    --policy.vision_encoder_pretrained_path="$VISION_PRETRAINED_PATH" \
    --policy.vision_mlp_projection_dim=$VISION_PROJECTION_DIM \
    --policy.vision_mlp_hidden_dim=$MLP_HIDDEN_DIM \
    --policy.vision_mlp_dropout=$MLP_DROPOUT \
    --policy.value_key="$VALUE_KEY" \
    --policy.value_dim=1 \
    --policy.value_bootstrap_steps="$N_STEP" \
    --policy.value_discount="$DISCOUNT" \
    --policy.value_reward_key=annotation_reward \
    --policy.value_target_tau="$TAU" \
    --steps=$STEPS \
    --policy.optimizer_lr=$LR \
    --policy.scheduler_warmup_steps=$WARMUP_STEPS \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.drop_proprioception_input=$DROP_PROPRIOCEPTION_INPUT \
    --policy.blackout_front_camera_input=$BLACKOUT_FRONT_CAMERA_INPUT \
    --policy.input_dropout_percent=$INPUT_DROPOUT_PERCENT \
    --policy.device=cuda \
    --batch_size=128 \
    --test_batch_size=128 \
    --test_freq=20 \
    --test_first_step=true \
    --test_frame_stride=10 \
    --log_freq=20 \
    --log_first_step=true \
    --save_freq=0 \
    --save_best_test_checkpoint=true \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}'
