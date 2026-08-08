#!/bin/bash
#SBATCH --job-name=mlp_success
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
LOSS=${LOSS:-l2}
CAMERAS=${CAMERAS:-side,wrist,front}
LR=${LR:-1e-5}
TEST_DATASET=${TEST_DATASET:-walle_skywalker_testset}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DROP_PROPRIOCEPTION_INPUT=${DROP_PROPRIOCEPTION_INPUT:-true}
VISION_PROJECTION_DIM=${VISION_PROJECTION_DIM:-256}
MLP_HIDDEN_DIM=${MLP_HIDDEN_DIM:-512}
MLP_DROPOUT=${MLP_DROPOUT:-0.1}
FREEZE_VISION_ENCODER=${FREEZE_VISION_ENCODER:-true}
SEED=${SEED:-1000}
if [ "$LOSS" != "l1" ] && [ "$LOSS" != "l2" ]; then
    echo "LOSS must be 'l1' or 'l2', got '$LOSS'" >&2
    exit 1
fi
SELECTED_IMAGE_KEYS="["
SELECTED_CAMERAS=""
for CAMERA in ${CAMERAS//,/ }; do
    case "$CAMERA" in
        side) IMAGE_KEY=observation.images.camera_side ;;
        wrist) IMAGE_KEY=observation.images.camera_wrist ;;
        front) IMAGE_KEY=observation.images.camera_front ;;
        *)
            echo "Unknown camera '$CAMERA' (expected a combination of side, wrist, and front)" >&2
            exit 1
            ;;
    esac
    if [[ ",$SELECTED_CAMERAS," == *",$CAMERA,"* ]]; then
        echo "CAMERAS must not contain duplicates, got '$CAMERAS'" >&2
        exit 1
    fi
    if [ -n "$SELECTED_CAMERAS" ]; then
        SELECTED_CAMERAS+=","
        SELECTED_IMAGE_KEYS+=","
    fi
    SELECTED_CAMERAS+="$CAMERA"
    SELECTED_IMAGE_KEYS+="\"$IMAGE_KEY\""
done
if [ -z "$SELECTED_CAMERAS" ]; then
    echo "CAMERAS must include at least one of side, wrist, or front" >&2
    exit 1
fi
SELECTED_IMAGE_KEYS+="]"
if [ "$SEED" != "1000" ]; then
    JOB_NAME="${JOB_NAME}_seed_${SEED}"
fi
PALIGEMMA_PRETRAINED_PATH=google/paligemma-3b-pt-224
PI05_BASE_PRETRAINED_PATH=/coc/testnvme/jcoholich3/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30
OUTDIR=./outputs/$JOB_NAME
DATASET='plug5_offline_rl_dataset'
DATA_ROOT=/coc/testnvme/jcoholich3/lerobot_data

echo "Job name: $JOB_NAME"
echo "Output dir: $OUTDIR"
echo "Target key: sparse_returns_gamma_1.0"
echo "Regression loss: $LOSS"
echo "Cameras: $SELECTED_CAMERAS"
echo "Camera sampling: each selected camera contributes independent single-image samples"
echo "Vision initialization: $INIT"
echo "Learning rate: $LR"
echo "Test dataset: $TEST_DATASET"
echo "Weight decay: $WEIGHT_DECAY"
echo "Drop proprioception input: $DROP_PROPRIOCEPTION_INPUT"
echo "Vision projection dim: $VISION_PROJECTION_DIM"
echo "MLP hidden dim: $MLP_HIDDEN_DIM"
echo "MLP dropout: $MLP_DROPOUT"
echo "Freeze vision encoder: $FREEZE_VISION_ENCODER"
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
    --test_dataset.repo_id=$TEST_DATASET \
    --test_dataset.root="$DATA_ROOT/$TEST_DATASET" \
    --policy.type=pi05 \
    --seed=$SEED \
    --output_dir=$OUTDIR \
    --job_name=$JOB_NAME \
    --policy.repo_id=your_repo_id \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=success_prediction \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=$FREEZE_VISION_ENCODER \
    --policy.train_expert_only=false \
    --policy.use_value_model=true \
    --policy.value_backbone=vision_mlp \
    --policy.vision_encoder_pretrained_path="$VISION_PRETRAINED_PATH" \
    --policy.vision_mlp_projection_dim=$VISION_PROJECTION_DIM \
    --policy.vision_mlp_hidden_dim=$MLP_HIDDEN_DIM \
    --policy.vision_mlp_dropout=$MLP_DROPOUT \
    --policy.value_key=sparse_returns_gamma_1.0 \
    --policy.value_loss="$LOSS" \
    --selected_image_keys="$SELECTED_IMAGE_KEYS" \
    --policy.value_dim=1 \
    --steps=3000 \
    --policy.optimizer_lr=$LR \
    --policy.scheduler_warmup_steps=3000 \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.drop_proprioception_input=$DROP_PROPRIOCEPTION_INPUT \
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
