Start two policy servers, one for the task policy and one for the reset policy. The servers start empty, then lazily initialize the first policy requested.

```bash
python -m lerobot.async_inference.policy_server  --port=8080
python -m lerobot.async_inference.policy_server  --port=8081
```
Then just alternate between these two commands:

```
bash pi_05_server_inference.bash \
  --prompt "Plug the charger into the power strip" \
  --name plug3_bc_and_dagger \
  --checkpoint 3000 \
  --port 8080 \
  --record "name of recording"

bash pi_05_server_inference.bash \
  --prompt "Unplug the charger" \
  --name plug3_bc_and_dagger \
  --checkpoint 3000 \
  --port 8081 \
  --record "name of recording"


```
