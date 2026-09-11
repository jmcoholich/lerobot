# Persistent π0.5 policy server

From the repository root, activate the same Python environment you normally use
for `collect_eval.bash`. In one terminal, start the empty server:

```bash
bash pi_05_policy_server.bash
```

Wait for `Empty policy server listening`. In another terminal, run as usual:

```bash
bash collect_eval.bash trial_01
# Stop the rollout with Ctrl+C, then start another:
bash collect_eval.bash trial_02
```

You can also run `bash pi_05_inference.bash trial_01` directly when you manage
OpenTeach data collection separately. Reusing a recording name overwrites its
trajectory HDF5 file when the new rollout starts recording.

Keep the checkpoint, task, and inference settings in `pi_05_inference.bash`:
`--policy.path`, `--policy.dtype`, `--policy.n_action_steps`, and any other
`--policy.*=VALUE` overrides are sent to and resolved by the server. The task is
`--task`, the rollout time limit is `--duration` (seconds), and `--fps` defaults
to 30. No shared rollout config file is needed, and `collect_eval.bash` is unchanged.

Set the intervention mode in `pi_05_inference.bash` before each rollout:

| Mode | `--interventions` | `--manual_guidance` | `--vis_spreads` |
| --- | --- | --- | --- |
| No guidance (default) | `none` | `false` | `false` |
| PIVOT | `PIVOT` | `false` | `false` |
| Primitive | `primitive` | `false` | `false` |
| Ensemble | `ensemble` | `false` | `false` |
| Manual guidance | `none` | `true` | `false` |
| Visualize trajectory spreads | `none` | `false` | `true` |

Restart the server once to load this implementation. Afterward, these settings
take effect on the next rollout without restarting the server or reloading
weights. Conflicting modes are rejected. Automatic strategies intervene when
`MMD² + 5 * diversity > 1.2082`, using zero for undefined MMD. Each trajectory
chunk records the score, threshold decision, configured setting, and whether
guidance occurred. Manual guidance uses the existing debugger
interaction in the server terminal. Candidate trajectories remain recorded in
all modes.

The first automatic intervention rollout initializes the VLM client and requires
the VLM service supplied for each invocation:
`bash pi_05_inference.bash RECORD_NAME http://HOST:PORT`. The second argument is
required and is forwarded as `--vlm_server_url`; there is no default URL. Supply only the base URL; the client adds `/health` and
`/v1/chat/completions`. Later strategy switches reuse that client; changing the
URL creates a new client on the next automatic intervention rollout. The effective mode is logged and saved in trajectory metadata,
including `rollout_config.intervention_settings`. Editing Python source or other
module constants such as `USE_WRIST` still requires restarting the server.

The launcher uses `lerobot.scripts.pi05_inference`, a Franka inference-only client.
It does not create, delete, or write a LeRobot dataset. OpenTeach camera recording,
Franka observation/command history, and server-side trajectory recording remain
enabled. Model configuration parsing, feature conversion, preprocessing, and
inference run on the server, keeping the training/recording dependencies out of
the client process. Restart the server after switching to this client.

The first rollout loads the model and processors. Subsequent runs log
`Reusing cached policy` when the resolved policy configuration, dataset features,
statistics, and rename map match. Changing the checkpoint or policy settings
reloads the model; changing the task, intervention settings, recording name, or trajectory directory
does not. The server retains one model at a time. Restart it if you replace
checkpoint files in place or edit model code. Python imports, robot startup,
and per-recording trajectory metadata still take place. The checkpoint SHA-256
is computed once per loaded model, before its first action chunk, and reused
across rollouts. The server logs when that initial hash starts and finishes.

To locate remaining delays, the client logs `Startup:` durations for robot
and policy/processor setup, followed by `First action:` camera/state
acquisition and inference/command durations. Python imports happen before these
setup logs. The server logs trajectory metadata setup and `First chunk:` times
for sampling 15 candidates, selecting trajectories, and writing them to disk.
Candidate generation and recording still run when interventions are disabled.
These are wall-clock phase timings, not GPU kernel profiling; camera streaming
and a live robot are needed to measure the complete startup.

Each rollout resets TACO's chunk counter, queued actions, replay state, and
processor state. Trajectories use the current client's `LEROBOT_DEMO_NAME` and
`LEROBOT_TRAJECTORY_DIR`, including the directory set by `collect_eval.bash`.
Inference, VLM calls, visualizations, and any manual guidance interaction run
in the server terminal; robot commands and observation/command history stay in the client.
Replay metadata retains the `rollout_config.dataset` task/FPS/time-limit fields
for existing analysis tools; those fields do not enable LeRobot dataset recording.
TACO's `MAX_CHUNKS` exit ends the client rollout while keeping the server alive.
Ctrl+C in the rollout terminal also leaves the server alive. An in-progress
inference call may finish before the server accepts the next rollout.

Ctrl+C in the **server** terminal stops the server and releases the model.

The transport is a Unix socket restricted to your user on this machine, with
one active rollout at a time. To change its path, set the same value in both
terminals before launching:

```bash
export LEROBOT_POLICY_SERVER=/tmp/my-pi05.sock
```

After an unclean server kill (for example, `kill -9`), remove its stale socket
only after checking the old server is stopped. The default path is
`/tmp/lerobot-pi05-${UID}.sock`. This lightweight client requires the server.
The separate `lerobot_record.py` entrypoint remains available for workflows that
need LeRobot dataset recording or local policy execution, using its original
`--dataset.*` arguments.

`main` already implements empty-server startup and checkpoint reuse in
`lerobot.async_inference.policy_server`, driven by `pi_05_server_inference.bash`.
That async client calls `predict_action_chunk` directly. This branch's server
instead uses the existing synchronous `predict_action` / `select_action` path
to preserve TACO candidate selection, guidance, and trajectory recording.

Set `--vlm_model_name` in `pi_05_inference.bash` to the model identifier returned
by the VLM service’s `/v1/models` endpoint. Changing the model name recreates the
VLM client on the next automatic intervention rollout without reloading policy weights.
