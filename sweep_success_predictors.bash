#!/bin/bash
#SBATCH --job-name=success_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-179%60
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --qos=short
#SBATCH --exclude=ig-88,megazord,cyborg,sonny,spd-13

set -euo pipefail
export LC_ALL=C

SWEEP_VERSION=success_prediction_v1
SEEDS=(1000 2000)
CAMERA_OPTIONS=(front side wrist side,wrist,front)
AUGMENTATION_OPTIONS=(none mild strong)
LR_OPTIONS=(3e-6 1e-5 3e-5)
LOSS_OPTIONS=(l1 l2)
MLP_DROPOUT_OPTIONS=(0.0 0.1 0.3)
MODEL_VARIANTS=(smolvlm mlp_frozen mlp_unfrozen)

select_variant_configs() {
    local variant=$1
    local camera augmentation lr loss dropout config hash

    for augmentation in "${AUGMENTATION_OPTIONS[@]}"; do
        for loss in "${LOSS_OPTIONS[@]}"; do
            {
            for camera in "${CAMERA_OPTIONS[@]}"; do
            for lr in "${LR_OPTIONS[@]}"; do
                if [ "$variant" = smolvlm ]; then
                    config="$variant|$camera|$augmentation|$lr|none|$loss"
                    hash=$(printf '%s' "$SWEEP_VERSION|selection|$config" | md5sum)
                    printf '%s|%s\n' "${hash%% *}" "$config"
                else
                    for dropout in "${MLP_DROPOUT_OPTIONS[@]}"; do
                        config="$variant|$camera|$augmentation|$lr|$dropout|$loss"
                        hash=$(printf '%s' "$SWEEP_VERSION|selection|$config" | md5sum)
                        printf '%s|%s\n' "${hash%% *}" "$config"
                    done
                fi
            done
            done
            } | sort | head -n 5 | cut -d'|' -f2-
        done
    done
}

CONFIGS=()
for variant in "${MODEL_VARIANTS[@]}"; do
    mapfile -t variant_configs < <(select_variant_configs "$variant")
    CONFIGS+=("${variant_configs[@]}")
done

# Expand each configuration to both seeds, then shuffle all 180 jobs so paired
# seeds and related configurations are not adjacent in the Slurm array.
mapfile -t JOBS < <(
    for config in "${CONFIGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job="$config|$seed"
            hash=$(printf '%s' "$SWEEP_VERSION|order|$job" | md5sum)
            printf '%s|%s\n' "${hash%% *}" "$job"
        done
    done | sort | cut -d'|' -f2-
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array job}
if (( TASK_ID < 0 || TASK_ID >= ${#JOBS[@]} )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#JOBS[@]} - 1)), got $TASK_ID" >&2
    exit 1
fi

IFS='|' read -r variant cameras augmentation lr dropout loss seed <<< "${JOBS[$TASK_ID]}"
camera_name=${cameras//,/_}
launcher_job_name="success_${variant}_${camera_name}_${augmentation}_${loss}_lr${lr}_drop${dropout}"
launcher_job_name=${launcher_job_name//./p}
job_name=$launcher_job_name
if [ "$seed" != 1000 ]; then
    job_name="${job_name}_seed_${seed}"
fi

case "$augmentation" in
    none)
        AUGMENTATION_ARGS=(--dataset.image_transforms.enable=false)
        ;;
    mild)
        AUGMENTATION_ARGS=(
            --dataset.image_transforms.enable=true
            --dataset.image_transforms.max_num_transforms=2
            --dataset.image_transforms.random_order=true
            '--dataset.image_transforms.tfs={brightness: {type: ColorJitter, kwargs: {brightness: [0.9, 1.1]}}, contrast: {type: ColorJitter, kwargs: {contrast: [0.9, 1.1]}}, saturation: {type: ColorJitter, kwargs: {saturation: [0.9, 1.1]}}, hue: {type: ColorJitter, kwargs: {hue: [-0.02, 0.02]}}, affine: {type: RandomAffine, kwargs: {degrees: [-3.0, 3.0], translate: [0.03, 0.03]}}}'
        )
        ;;
    strong)
        AUGMENTATION_ARGS=(
            --dataset.image_transforms.enable=true
            --dataset.image_transforms.max_num_transforms=4
            --dataset.image_transforms.random_order=true
            '--dataset.image_transforms.tfs={brightness: {type: ColorJitter, kwargs: {brightness: [0.6, 1.4]}}, contrast: {type: ColorJitter, kwargs: {contrast: [0.6, 1.4]}}, saturation: {type: ColorJitter, kwargs: {saturation: [0.5, 1.5]}}, hue: {type: ColorJitter, kwargs: {hue: [-0.1, 0.1]}}, sharpness: {type: SharpnessJitter, kwargs: {sharpness: [0.3, 1.8]}}, affine: {type: RandomAffine, kwargs: {degrees: [-10.0, 10.0], translate: [0.1, 0.1], scale: [0.9, 1.1]}}}'
        )
        ;;
    *)
        echo "Unknown augmentation preset: $augmentation" >&2
        exit 1
        ;;
esac

echo "Sweep task: $TASK_ID/${#JOBS[@]}"
echo "Configuration: ${JOBS[$TASK_ID]}"
echo "Job name: $job_name"

if [ "$variant" = smolvlm ]; then
    LOSS=$loss CAMERAS=$cameras LR=$lr SEED=$seed \
        bash pi05_train_success_predictor.bash "$launcher_job_name" "${AUGMENTATION_ARGS[@]}"
else
    if [ "$variant" = mlp_frozen ]; then
        freeze_vision_encoder=true
    else
        freeze_vision_encoder=false
    fi
    LOSS=$loss CAMERAS=$cameras LR=$lr SEED=$seed INIT=pi05 MLP_DROPOUT=$dropout \
        FREEZE_VISION_ENCODER=$freeze_vision_encoder \
        bash train_MLP_success_predictor.bash "$launcher_job_name" "${AUGMENTATION_ARGS[@]}"
fi
