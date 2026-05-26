Start two policy servers, one for the task policy and one for the reset policy. The servers start empty, then lazily initialize the first policy requested.

```bash
python -m lerobot.async_inference.policy_server --port 8080
python -m lerobot.async_inference.policy_server --port 8081
```
Then just alternate between these two commands:

```
bash pi_05_server_inference.bash \
  --prompt "Plug the charger into the power strip" \
  --name plug5 \
  --chunk-size 40 \
  --checkpoint 3000 \
  --port 8080 \
  --record "plug5_rollout"

bash pi_05_server_inference.bash \
  --prompt "Unplug the charger" \
  --name unplug5 \
  --chunk-size 40 \
  --checkpoint 3000 \
  --port 8081 \
  --record "unplug5_rollout"
```

Or run the alternation automatically:

```bash
bash auto_rollouts.bash
```
