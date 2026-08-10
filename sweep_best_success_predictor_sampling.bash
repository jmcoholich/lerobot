#!/bin/bash
#SBATCH --job-name=success_sampling
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-23%24
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --qos=short
#SBATCH --exclude=ig-88,megazord,cyborg,sonny,spd-13

set -euo pipefail
export LC_ALL=C

SEEDS=(1000 2000 3000)
CAMERA_OPTIONS=(wrist side,wrist,front)
EQUAL_WEIGHT_OPTIONS=(false true)
REJECT_LONG_OPTIONS=(false true)

# Frozen PI05-initialized MLP, mild augmentation, LR 3e-5, dropout 0.1, and L2
# was the best configuration in the previous sweep. Vary only the requested
# camera and episode-sampling settings, with three seeds per configuration.
mapfile -t JOBS < <(
    for cameras in "${CAMERA_OPTIONS[@]}"; do
        for equal_weight in "${EQUAL_WEIGHT_OPTIONS[@]}"; do
            for reject_long in "${REJECT_LONG_OPTIONS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    job="$cameras|$equal_weight|$reject_long|$seed"
                    hash=$(printf '%s' "success_sampling_v1|$job" | md5sum)
                    printf '%s|%s\n' "${hash%% *}" "$job"
                done
            done
        done
    done | sort | cut -d'|' -f2-
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array job}
if (( TASK_ID < 0 || TASK_ID >= ${#JOBS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#JOBS[@]} - 1)), got $TASK_ID" >&2
    exit 1
fi

IFS='|' read -r cameras equal_weight reject_long seed <<< "${JOBS[$TASK_ID]}"
camera_name=${cameras//,/_}
job_name="success_mlp_frozen_${camera_name}_mild_l2_lr3e-5_drop0p1_equaltraj_${equal_weight}_reject1300_${reject_long}"
job_name=${job_name//./p}

AUGMENTATION_ARGS=(
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=2
    --dataset.image_transforms.random_order=true
    '--dataset.image_transforms.tfs={brightness: {type: ColorJitter, kwargs: {brightness: [0.9, 1.1]}}, contrast: {type: ColorJitter, kwargs: {contrast: [0.9, 1.1]}}, saturation: {type: ColorJitter, kwargs: {saturation: [0.9, 1.1]}}, hue: {type: ColorJitter, kwargs: {hue: [-0.02, 0.02]}}, affine: {type: RandomAffine, kwargs: {degrees: [-3.0, 3.0], translate: [0.03, 0.03]}}}'
)

echo "Sweep task: $TASK_ID/${#JOBS[@]}"
echo "Configuration: ${JOBS[$TASK_ID]}"
echo "Job name: $job_name"

LOSS=l2 CAMERAS=$cameras LR=3e-5 MLP_DROPOUT=0.1 INIT=pi05 \
    FREEZE_VISION_ENCODER=true SEED=$seed \
    EQUAL_WEIGHT_TRAJECTORIES=$equal_weight REJECT_EPISODES_OVER_1300=$reject_long \
    bash train_MLP_success_predictor.bash "$job_name" "${AUGMENTATION_ARGS[@]}"
