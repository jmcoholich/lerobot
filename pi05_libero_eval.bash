HF_CACHE='/home/jcoholich/.cache/huggingface'

lerobot-eval \
   --env.type=libero \
   --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
   --eval.batch_size=1 \
   --eval.n_episodes=2 \
   --policy.path=lerobot/pi05_libero_base \
   --policy.n_action_steps=10 \
   --output_dir=./eval_logs/test5_pi05_libero_base \
   --env.max_parallel_tasks=1 \