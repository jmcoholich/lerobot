# TACO trajectory recordings and offline replay

`bash collect_eval.bash asdf` writes
`/home/jeremiah/openteach/extracted_data/demonstration_asdf/trajectories_asdf.h5`.
The inference launcher exports `LEROBOT_DEMO_NAME`; `LEROBOT_TRAJECTORY_DIR`
still overrides the directory. Reusing a demo name overwrites its trajectory HDF5
file when the new rollout starts recording.

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
| `temporal_mmd/mmd2`, `temporal_mmd/gamma` | Online squared RBF MMD against the previous chunk and the effective kernel parameter |
| `candidate_diversity/score`, `candidate_diversity/gamma` | One minus mean within-current-chunk RBF similarity and its effective gamma |

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
The rollout chunk limit is `800 // n_action_steps`, including manual guidance:
8 chunks at 100 steps, 16 at 50, or 32 at 25. Only complete execution chunks are
queued, so horizons that do not divide 800 stop below the 800-action cap.
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

## Rollout and chunk timing

New rollouts save durations in seconds, measured with monotonic
`time.perf_counter()`. Wall-clock `*_at_unix` values are provided for correlation
with other logs; durations never subtract clocks from different processes.

Each `chunks/<index>/timing` group contains server-side measurements:

| Dataset | Measured interval |
| --- | --- |
| `base_inference_s` | Generate the 15 unperturbed candidates, including input preparation and recorded CPU snapshots; measured on every chunk |
| `guidance_selection_s` | Guidance selection/construction, including VLM calls, prompt images, ensemble blending/visualization, and manual debugger waits |
| `guided_inference_s` | Additional model inference using the selected guidance, including its CPU snapshot |
| `intervention_s` | `guidance_selection_s + guided_inference_s`; zero when no guidance branch runs |
| `policy_generation_s` | Whole server chunk-generation path, including base inference, candidate selection, initial HDF5 recording, and any intervention; ends before the execution-decision HDF5 write |

Sampling snapshots copy outputs to CPU, so inference durations include waiting
for those GPU results. They measure wall time, not isolated GPU kernel time.
For inference without interventions, use `base_inference_s`, or filter chunks
with `intervention/occurred=False` and use `policy_generation_s` for their whole
generation path. These intervals overlap; do not sum all timing fields.

The `pi05_inference.py` client adds the following fields after a completed chunk
or during graceful shutdown of a partially executed chunk:

| Dataset | Measured interval |
| --- | --- |
| `chunk_total_s` | First observation request through the last completed loop wait (or partial-chunk shutdown), including inference and intervention latency |
| `execution_s` | First `send_action` start through that same endpoint, including subsequent observations, queued-action RPCs, command sends, and scheduled waits; excludes initial chunk-generation latency |
| `command_send_s` | Sum of time inside `robot.send_action` calls that returned successfully |
| `observation_s`, `inference_rpc_s` | Accumulated successful observation and policy-request times, respectively |
| `executed_actions`, `execution_complete` | Number of successfully returned command sends and whether all queued actions plus their loop waits completed |

`execution_s` measures the command loop, not an independently detected physical
settling time. `started_at_unix`, `execution_started_at_unix`, and `ended_at_unix`
record its boundaries. On an interrupted operation, total durations include the
wait until shutdown while successful-call subtotals may exclude the interrupted call.

The visualizer displays `Execution + inference` on every chunk as
`execution_s + base_inference_s`. It excludes intervention time and shows
`Intervention time` separately from `intervention_s` only when guidance occurred.
Durations use three decimal places in seconds. Missing measurements show
`not recorded`, invalid values show `N/A`, and partial execution is labeled.

Root `/timing/elapsed_s` measures the rollout from after robot connection and
policy reset to loop termination, excluding setup/model loading and robot
disconnect/history saving. It includes observation, inference, interventions,
execution, scheduled waits, and timing-report overhead. The root group also
records start/end/update timestamps, `executed_actions`, `completed_chunks`,
`finalized`, and an `end_reason` attribute. It is updated after every completed
chunk and finalized at normal duration/step limits or graceful interruption.
Chunk reports are written by the server; the client never opens the HDF5 file.

An interrupted RPC, lost connection, or abrupt process kill can prevent the
final report; prior completed-chunk reports remain readable. Treat missing
timings or `finalized=False` as incomplete, not as a final rollout duration.
The persistent server accepts final timing reports after returning a rollout
stop or error. Restart the policy server once when installing this change.

Old trajectory logs only had a chunk timestamp, and Deoxys history has command
timestamps without inference/intervention labels. These detailed durations
cannot be reliably reconstructed from those timestamps alone. Existing HDF5
files are not backfilled by this change.

## Online overlap MMD

Every new chunk compares all 15 unperturbed candidate sequences with the previous
chunk's candidates. With prediction horizon 100 and execution horizon 50, it uses
`previous[:, 50:100, :7]` and `current[:, :50, :7]` from the first sampler calls.
Each overlap is flattened to `[15, 350]`: samples are whole sequences, preserving
temporal order. The seven normalized XYZ and quaternion dimensions are included;
the gripper and 24 padded dimensions are excluded. Selection, perturbations,
and guided samples do not enter this base-policy consistency score.

The statistic is the **biased squared MMD**, including kernel diagonals, matching
the reference code and [EVE Appendix E](https://arxiv.org/html/2512.21430v1#A5).
It uses `k(x,y) = exp(-gamma * ||x-y||²)`, the scikit-learn convention in the
reference's executable code (its docstring's extra `/2` is inconsistent).
Equivalently, `gamma = 1 / (2 * sigma²)` for Gaussian width `sigma`.

`pi_05_inference.bash` exposes `--policy.mmd_gamma` through `LEROBOT_MMD_GAMMA`:

```bash
bash collect_eval.bash asdf                         # adaptive median default
LEROBOT_MMD_GAMMA=0.002 bash collect_eval.bash trial # example fixed gamma, not calibrated
```

- `median`: recompute `gamma = 1 / (2 * median(positive squared distances))`
  over pooled previous/current sequence samples at each chunk. Start here.
- A positive finite number: hold gamma fixed across chunks and rollouts.
  Larger gamma makes the kernel more sensitive to small differences.
- `max_eig`: use the reciprocal of the largest pooled sample-covariance eigenvalue,
  matching the alternate reference heuristic. Prefer `median` initially.

For a fixed setting, collect representative successful rollouts with `median`,
take the median of their finite saved gammas as an initial value, and compare
`0.25x`, `1x`, and `4x` that value on held-out rollouts. Choose a setting that
separates relevant inconsistencies without saturating scores. Recalibrate after
changing the action normalization, overlap horizon, or sample count. Adaptive
gamma changes the distance scale each chunk; fixed gamma gives a common kernel
for absolute comparisons. The existing 15 samples add no inference calls, but
provide a noisier distribution estimate than larger batches.
[Sentinel Appendix A.1](https://arxiv.org/pdf/2410.04640) discusses bandwidth and
sample-count selection. Intervention thresholds need calibration on your
rollouts; the papers' numerical thresholds do not transfer automatically.

Scores are written before execution to
`chunks/000001/temporal_mmd/mmd2` in `trajectories_<demo>.h5` and logged to the
server console. The group also stores `gamma`, `overlap_steps`, `num_samples`,
`action_dim`, and `previous_record_index`, with kernel/estimator/source attributes.
The first chunk after startup or reset, and chunks with no prediction overlap,
have `NaN` for score and effective gamma. Identical constant samples yield zero
with adaptive gamma falling back to 1. The requested setting is recorded in
`metadata_json.policy_config.mmd_gamma` and each group's `gamma_setting` attribute.

## Intervention trigger and recording

Automatic PIVOT, primitive, and ensemble guidance runs when
`MMD² + 5 * diversity > 1.715`, replacing the first-two-chunks schedule.
An undefined MMD (first chunk, reset, or no overlap) contributes zero to this
score; the saved MMD itself remains NaN. The first chunk is not forced to
intervene. Both metrics exclude gripper. These constants were calibrated on
20 rollouts with 50-step execution and produced a pooled intervention rate of
14.98% (46/307 chunks), including every first chunk, using full-horizon diversity.
Changing the execution horizon or metric gamma settings also requires recalibration.

The recorder uses the explicit execution horizon (`n_action_steps`) to align
overlaps, even when recorded candidates retain the full prediction horizon.
Each chunk records `execution_horizon` as an attribute and an `intervention`
group containing datasets `score`, `diversity_weight`, `threshold`, `triggered`,
and `occurred`. The group's `setting` attribute is `none`, `PIVOT`, `primitive`,
`ensemble`, `manual`, or `vis_spreads`.

For automatic interventions, this group also records the selected `pivot_color`
and/or `primitive` as attributes. These identify the choices actually used for
guidance, including the fallback primitive if the VLM returned an invalid label.
The visualization displays the PIVOT color, the primitive name, or both for EVE
beneath the scores. Older intervention records without these attributes display
`not recorded` for the missing choices.

`triggered` records whether the score exceeds the threshold, even when automatic
interventions are disabled. `occurred` becomes true only after sampling with
non-None guidance completes and the resulting action chunk is recorded for
execution. It does not confirm physical execution of every action. Check
`execution_ready` to distinguish completed decisions from interrupted chunks.
Manual guidance and spread visualization retain their existing behavior;
visualization alone is not counted as a guidance intervention.

PIVOT, primitive, and EVE guidance spans the full predicted action chunk
(`chunk_size=100`), independently of `n_action_steps`. EVE blends and plots
the full guidance trajectories. Reconstruction guidance requires exactly
`chunk_size` timesteps and pads only action dimensions, so every predicted
timestep receives a guidance residual. The primitive generator currently
produces 100 steps. Only the first `n_action_steps` of the guided prediction
are queued for execution. The overlap-based intervention trigger still depends
on `n_action_steps`; changing execution length can change when guidance triggers.

## Candidate diversity

`compute_mmd_rbf(current, previous, gamma)` returns `(mmd2, effective_gamma)`.
`compute_diversity_rbf(current, gamma)` returns `(diversity, effective_gamma)`.
The recorder computes diversity with its own fixed gamma. Its
formula is one minus the within-current term of the biased MMD estimator:
`1 - mean(k(current, current))`, including diagonal self-similarities. It uses
only candidates at the current observation; MMD uses both adjacent overlaps.

The recorder uses all 100 predicted steps of the same 15 unperturbed, normalized
candidates, independently of `n_action_steps`. MMD still uses the overlap. Both metrics retain all seven XYZ and quaternion dimensions and
exclude the final gripper dimension.
The score is saved under `chunks/<index>/candidate_diversity/score`, along with
`gamma`, `horizon`, `num_samples`, `action_dim`, and source/estimator attributes.
`--policy.diversity_gamma` is a positive, fixed number, independent of
`--policy.mmd_gamma`. The launcher exposes it through `LEROBOT_DIVERSITY_GAMMA`.
The default is **0.06302211495246096**, the median of the 11 finite
`temporal_mmd/gamma` values from complete chunks in
`/home/jeremiah/openteach/extracted_data/demonstration_try_for_success_50_5/trajectories_try_for_success_50_5.h5`.
That reference has no diversity fields and used a 100-step prediction horizon
and 50-step execution horizon, with all eight action dimensions. The first MMD
gamma is undefined and was excluded. Excluding the gripper does not change this
fixed gamma.
The value is stored in configuration, so rollouts do not reread the reference.
The same fixed gamma is used for every chunk, including the first after reset.
With no overlap, MMD remains undefined; diversity still uses the full predicted horizon.

For B candidates the score lies between 0 and 1 - 1/B. Identical candidates give
0; increasingly separated candidates approach 1 - 1/B at a fixed gamma. Higher
means more diverse. The fixed gamma gives a common distance
scale when comparing chunks. Keep the candidate count, action
normalization, and measured horizon consistent for such comparisons.

The visualizer displays `Candidate diversity` to three decimal places above each
image. Logs without this field display `not recorded`; existing HDF5 files are
not backfilled automatically. Earlier recordings using `-mean(k(current,current))`
retain their original values; their saved `estimator` attribute identifies that formula.
Earlier diversity scores that included the gripper also retain their saved values;
new recordings identify its exclusion in `source` and record `action_dim=7`.

The visualizer also shows `Intervention: <type>` in an amber badge when the
chunk's saved `intervention/occurred` flag is true. Ensemble guidance is labeled
`EVE`; other types retain their names. Chunks with no intervention show `none`,
and older logs without this flag show `not recorded`. A threshold trigger alone
does not produce a badge.

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

This writes `trajectories_asdf.png` beside the HDF5 file. Multiple paths and quoted
glob patterns are also accepted, with one PNG written beside each matching log:

```bash
python scripts/visualize_pi05_trajectories.py \
  '/home/jeremiah/openteach/extracted_data/demonstration_try_for_success_*/*trajector*.h5'
```

Patterns support recursive `**`. Non-HDF5 matches are ignored and duplicate paths
are rendered once. Use `*trajector*` to match the `trajectories_` filename prefix.
Use `--output path.png` to choose a different destination for a single matching
file. The image header displays the checkpoint,
intervention setting, and `n_action_steps` from the log's `metadata_json`;
it also shows the recording time in readable English in Atlanta's Eastern timezone
(`America/New_York`, automatically labeled EST or EDT for the recording date),
using the saved `created_at_utc` (or earliest chunk timestamp if absent), never
the image creation time. Missing fields are labeled "not recorded". Each tile shows the full predicted horizon
(100 steps for the current checkpoint) of the same five selected trajectories.
Each tile also displays its saved `MMD^2 vs previous` score. Undefined comparisons
(first chunk or no overlap) show `N/A`; logs without saved MMD show `not recorded`.
Green highlighting is normalized per episode over the displayed finite scores:
scores at or below the 20th percentile are unhighlighted, and scores at or above
the 80th percentile are solid green, with linear opacity in between. Displayed
MMD values remain unchanged. If the percentile limits coincide, all scores stay
unhighlighted. Undefined and missing scores also stay unhighlighted; text turns
dark on stronger highlights for contrast.
Candidate diversity uses blue highlighting, normalized separately between the
minimum and maximum finite diversity scores in each file: the minimum has no
highlight, the maximum is solid blue, and intermediate values use linear opacity.
Equal or missing diversity scores stay unhighlighted. Diversity text stays light
for contrast against blue; normalization uses the full-precision saved values.
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
