#!/bin/bash
#SBATCH --job-name=libero_eval
#SBATCH -p kira-lab
#SBATCH -A kira-lab
#SBATCH -G a40:1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH --qos=long

HF_CACHE='/home/jcoholich/.cache/huggingface'

export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# source /coc/testnvme/$USER/.bashrc
# conda activate lerobot

# sed -i 's|"/tmp/robosuite.log"|"/coc/testnvme/jcoholich3/lerobot/tmp/robosuite.log"|' \
#   /coc/testnvme/jcoholich3/miniforge3/envs/lerobot/lib/python3.10/site-packages/robosuite/utils/log_utils.py

# Xvfb :99 -screen 0 1280x1024x24 &
# XVFB_PID=$!
# export DISPLAY=:99

export QT_QPA_PLATFORM=offscreen
export QT_OPENGL=egl
export EGL_PLATFORM=surfaceless

# xvfb-run -a -s "-screen 0 1280x1024x24"
lerobot-eval \
   --env.type=libero \
   --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
   --eval.batch_size=1 \
   --eval.n_episodes=2 \
   --policy.path=jcoholich/pi05_droid_converted \
   --policy.n_action_steps=10 \
   --output_dir=./eval_logs/test_pi05_droid_converted\
   --env.max_parallel_tasks=1 \