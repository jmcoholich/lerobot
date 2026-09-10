# TACO trajectory recordings and offline replay

`bash collect_eval.bash asdf` writes
`/home/jeremiah/openteach/extracted_data/demonstration_asdf/trajectories_asdf.h5`.
The inference launcher exports `LEROBOT_DEMO_NAME`; `LEROBOT_TRAJECTORY_DIR`
still overrides the directory. An existing filename raises `FileExistsError`,
so use a new demo name for a new rollout.

Files use schema version 2. Each complete candidate set is saved before guidance
or execution, and the execution decision is saved before actions enter the queue.
SIGINT is deferred while an HDF5 write is in progress. `complete` marks a saved
candidate set; `execution_ready` marks a saved execution decision. Queued actions
are planned actions, not confirmation that every action reached the robot.

Each `chunks/000000` group contains:

| Location | Contents |
| --- | --- |
| `raw_observation/observation.state` | Proprioception before normalization, with batch dimension |
| `raw_observation` attribute `task_json` | Original task text |
| `observations` | Policy-input state, all three camera images, token IDs, and attention mask; integer and boolean dtypes are preserved |
| `original_actions`, `perturbed_actions`, `selected_actions`, `selected_indices` | Original 15 trajectories, perturbed 15, selected five, and their indices |
| `perturbation_noise` | Standard-normal XYZ perturbation draws, excluding the first timestep; multiply by the `perturb_std` attribute |
| `sampling/000000/inputs` | Exact resized model images, image masks, tokens, attention mask, full initial flow-matching `noise`, and any guidance tensors |
| `sampling/000000` attribute `settings_json` | Sample count, integration steps, guidance settings, and autocast settings |
| `sampling/000000/full_output` | Full padded sampler output before execution-horizon truncation |
| `queued_actions` | Normalized action chunk chosen for execution |
| `execution_source` attribute | Whether execution used original sample 0, selected sample 0, or the final guided sampling call |

The first sampling call generates the 15 candidates. Additional calls record
guided/manual sampling, including its own initial noise and the exact guidance
tensor. Offline replay does not need to repeat a VLM selection.

With chunk size 100, max action dimension 32, and `n_action_steps=50`, initial
candidate noise has shape `[15, 100, 32]`; candidate trajectories have shape
`[15, 50, 8]`. Recording only noise truncated to 50 steps would be insufficient.

Root `metadata_json` includes the effective policy config, checkpoint path and
SHA-256, Torch/Transformers/CUDA versions, and backend settings. New recordings
also include `rollout_config`: the rollout's robot and policy settings, the original task prompt in
`rollout_config.dataset.single_task`, FPS, episode limits, and dataset settings.
`policy_config.n_action_steps` holds the execution horizon.
The inference-only client retains `rollout_config.dataset` for task/FPS/time
settings but does not write a LeRobot dataset.
`launch` records the Python argv, interpreter, working directory, and selected
inference environment variables. `taco_settings` records the chunk limit,
perturbation scale, and wrist-camera guidance setting; intervention flags are
also saved at the root metadata level.

`artifacts` contains snapshots of the policy, recorder entrypoint, Franka adapter,
and both launch scripts, plus the actual preprocessor and
postprocessor JSON/safetensors files, including normalization statistics.
The model weights remain in the referenced checkpoint and must be retained.
Hashing the checkpoint adds a one-time read before the first sampling call.
The policy server caches that hash with the loaded model across rollouts;
starting a new recording does not reread the weights.

For quick inspection:

```python
import h5py
import json

with h5py.File(path, "r") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    print(metadata["checkpoint_file"])
    print(metadata["policy_config"]["n_action_steps"])
    print(metadata["rollout_config"]["dataset"]["single_task"])
    print(metadata["launch"]["argv"])
```

Existing recordings are not rewritten. They still contain checkpoint and policy
settings plus per-chunk task text; recordings made before this addition lack the
full `rollout_config` and `launch` metadata. These settings and recorded inputs
support offline action replay, not reconstruction of physical scene dynamics.

## Replaying a chunk

Use the same checkpoint, recorded config, code, and runtime for the closest
numerical agreement. Different GPUs, library versions, or nondeterministic CUDA
kernels can produce small numerical differences even with identical noise.
The helper verifies sampler outputs, candidate stages, and queued actions with
floating-point tolerances; the noise tensors are used directly, so no RNG seed
reset is needed.

Given a `PI05PolicyTaco` instance named `policy` loaded from that checkpoint with
the recorded configuration:

```python
from lerobot.policies.pi05.trajectory_replay import (
    load_recorded_processor,
    replay_trajectory_chunk,
)

path = "/home/jeremiah/openteach/extracted_data/demonstration_asdf/trajectories_asdf.h5"
queued = replay_trajectory_chunk(policy, path, chunk_index=0)
# queued: normalized [1, n_action_steps, action_dim], on CPU

postprocessor = load_recorded_processor(path)
first_robot_action = postprocessor(queued[:, 0].clone())
```

The effective configuration is `json.loads(file.attrs["metadata_json"])["policy_config"]`.
It can be decoded with `draccus.decode(PI05Config, config_dict)` and passed as
`config=` to `PI05PolicyTaco.from_pretrained(...)`. Match the recorded backend
settings (`matmul_allow_tf32`, `cudnn_allow_tf32`, and `deterministic_algorithms`)
as well. Keep the checkpoint weights matching the recorded SHA-256.

Calling `predict_action_chunk(batch, noise=...)` also accepts explicit initial
noise for experiments. The replay helper uses the lower-level sampler with
the archived model inputs to bypass tokenizer and image-preprocessing changes.

## Visualizing all chunks

From the repository root, in the inference environment:

```bash
python scripts/visualize_pi05_trajectories.py \
  /home/jeremiah/openteach/extracted_data/demonstration_asdf/trajectories_asdf.h5
```

This writes `trajectories_asdf.png` beside the HDF5 file. Use `--output path.png`
to choose a different destination. The image header displays the checkpoint,
intervention setting, and `n_action_steps` from the log's `metadata_json`;
it also shows the recording time in readable English in Atlanta's Eastern timezone
(`America/New_York`, automatically labeled EST or EDT for the recording date),
using the saved `created_at_utc` (or earliest chunk timestamp if absent), never
the image creation time. Missing fields are labeled "not recorded". Each tile shows the full predicted horizon
(100 steps for the current checkpoint) of the same five selected trajectories.
Their recorded perturbed prefix is preserved; the remaining steps come from
the corresponding original samples in `sampling/000000/full_output`. The old
logs contain no perturbation draws for steps beyond `n_action_steps`, so those
remaining steps are unperturbed. This does not change which five samples were selected.

The trajectories are plotted
on each chunk's front-camera observation, using `visualize_trajectories_on_camera()`
directly: identical calibration, action min/max conversion, colors, legend,
brightness, and 360-pixel crop. The script draws every trajectory point, including
step 99, using `trajectory_stride=1`; inference visualization still defaults to
stride 10. Trajectories use lines without point or start-position markers, and
the legend uses colored line swatches. Tiles are ordered by
chunk, labeled, and laid out six per row; unused slots in the last row stay blank.
Incomplete chunks are skipped. The script reads normalization statistics from
the HDF5 postprocessor artifact and requires a schema-2 log and the inference
Python dependencies. It does not load model weights or connect to the robot.
Rendering uses the current branch's camera calibration.
