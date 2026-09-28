# LeRobot regression audit: feat/pi05_kv_cache

The strongest code changes are the action execution horizon and synchronous recording/sampling work, but the evidence does **not** show a broken policy, normalization change, camera swap, stale camera frames, or KV-cache numerical regression. The broader investigation found a systematic block-layout reversal, which should rank above the code changes below as an explanation of repeatedly revisiting the first pickup site.

## Historical baseline and actual runtime

- Repository HEAD is `89dff82b` (2026-09-11); branch `feat/pi05_kv_cache`. The only pre-existing tracked modification is `pi_05_inference.bash`, changing `n_action_steps=75` to `100`; it was preserved.
- Demonstrations were collected March 23. The supplied successful `base_rollouts_original` records are May 9, according to their raw timestamps, not March. Thus `f6ee46ad` (May 7) is the last committed ancestor before those successful rollouts. This is a historical source baseline, not proof of the exact May 9 in-memory code: old rollouts do not include the source/runtime archives that new rollouts include.
- The May 7 launcher already selected the current checkpoint path and exact task text `place both blocks in the bin`. It did not override `n_action_steps`; the checkpoint config has `100`. It already used the `absolute_eef_pose_to_delta` Franka path.
- Every new `dracorex_base_0` through `_6` archive records `n_action_steps=75`, `chunk_size=100`, 10 denoising steps, bfloat16, no AMP, no RTC, no compilation, and interventions/manual guidance/spread visualization all disabled. Every rollout executed 13 chunks of 75 actions (975 total).
- All seven archives record checkpoint SHA-256 `b6d2fc2184bc0b44ebb423853dededf20d08578ac3b33604edd4db16b16a85ea`. Independently hashing the current 7.47 GB checkpoint yields that exact digest. The checkpoint directory/config/weights date to March 24. This establishes the new runs' provenance, but no historical hash exists here to prove the May 9 runs used identical bytes.
- Archived source files match the current files except the dirty launcher's 75-to-100 edit. Archives save source files as present on disk; those alone are not bytecode snapshots. The numerical replay below additionally validates executed sampler behavior.

Checkpoint: `/data3/lerobot_checkpoints/both_in_bin_interleaved/checkpoints/003000/pretrained_model`.
Archive example: `/data3/extracted_data/demonstration_dracorex_base_0/trajectories_dracorex_base_0.h5`, with `metadata_json` in root attributes, `artifacts/*`, `chunks/*`, and `timing/*`.

## Changes that actually affect base rollouts

1. **Execution horizon shortened from 100 to 75, with an intermediate 50-step setting.** Commit `d9384365` (Sept 9, 21:24 EDT) overrides it to 50. Commit `2109ff7d` (Sept 11, 15:44 EDT) changes it to 75. The recorded runs used 75 even though the working launcher now says 100. This changes when the robot observes/replans and discards the final 25 steps of each predicted 100-step chunk. It is a real closed-loop behavior change, but shorter horizons can improve or worsen behavior and the videos do not establish causation. Restore 100 for a controlled reproduction after restoring the demonstrated layout. Relevant code: [current launcher](/home/jeremiah/lerobot/pi_05_inference.bash:32), [chunk truncation](/home/jeremiah/lerobot/src/lerobot/policies/pi05/modelling_pi05_taco.py:1821).

2. **Base policy now samples 15 candidates and records them synchronously.** Commit `d9384365` changes the old one-sample unguided branch into an unconditional 15-candidate pipeline. It still executes the first **unperturbed** sample, not the selected/perturbed representative. This matters principally for latency and RNG consumption. Across all 91 archived chunks, `queued_actions` exactly equals `original_actions[0:1, :75]`; no perturbation or guidance leaked into the base actions. Relevant code: [candidate generation](/home/jeremiah/lerobot/src/lerobot/policies/pi05/modelling_pi05_taco.py:1648), [unconditional call](/home/jeremiah/lerobot/src/lerobot/policies/pi05/modelling_pi05_taco.py:1731), [base action selection](/home/jeremiah/lerobot/src/lerobot/policies/pi05/modelling_pi05_taco.py:1799).

   Recorded median sampling time is 391.6 ms; median full policy generation is 443.1 ms (range 425.4–473.1 ms), excluding later execution. Median chunk execution is 3.756 s and total chunk duration is 4.242 s. The robot therefore pauses roughly half a second at replanning. The old command-timestamp audit shows shorter gaps; see the separate dynamics report. Do not ascribe all of the gap increase to 15 samples: an offline same-runtime paired measurement yielded medians about 448 ms for batch 15 and 423 ms for batch 1, with substantial timing variability. The actual historical environment/runtime is unavailable.

3. **Persistent local policy server and RPC introduced Sept 10 (`7ba3d5e2`).** Every action, including queued actions, passes through a synchronous request/reply, and image/state processing still happens per action. This adds per-step overhead and transfers three camera arrays even when the policy needs no new chunk. The server resets action queues, counters, and processor state between rollouts; configuration changes, including horizon, are in its cache key. No evidence of stale queues or wrong checkpoint reuse was found. Relevant code: [client loop](/home/jeremiah/lerobot/src/lerobot/scripts/pi05_inference.py:91), [cache and reset](/home/jeremiah/lerobot/src/lerobot/scripts/pi05_policy_server.py:139).

4. **Stop condition changed from six chunks to a 1,000-action cap.** May 7 has `MAX_CHUNKS=6`; current code ends at `1000 // n_action_steps` chunks, hence 975 steps at horizon 75. This extends attempts; it does not explain failure to reach the second block. Relevant code: [cap](/home/jeremiah/lerobot/src/lerobot/policies/pi05/modelling_pi05_taco.py:1724).

The September intervention thresholds, new VLM model/prompt, ensemble weights, and disabled downward primitive do not activate in the supplied base rollouts because interventions are false. March 17/27 joint-action additions are disabled by `DELTA_JOINT_ACTIONS=False` and `JOINT_ACTIONS=False`. The May 7 Franka operator API update was already present before the successful May 9 records. Avoid using unrelated main-branch May 22 gripper commits as evidence that this branch changed after the successful runs.

## Numerical offline validation

Ran on the local RTX 4090 with network access disabled in model loading and no robot instantiation or commands. Selected chunks **0, 1, 3, 6, 10 from all seven new rollouts**: 35 observations, 15 candidate trajectories each, and 75 executed steps per chunk.

- Current model code + current checkpoint + archived model inputs + archived initial diffusion noise reproduced all 35 recorded sampler outputs **bit-for-bit**.
- For each same observation and the first candidate's exact same noise, reran the sampler with `num_samples=1`. Across 2,625 executed steps, mean Cartesian action difference was **0.127 mm**, maximum **0.534 mm**; maximum quaternion rotation difference was **0.123 degrees**.
- No sign/decision changes occurred in the gripper command at the controller's actual **zero threshold**. Floating command values can differ slightly, so this does not assert byte-identical actuator commands for batch 1.
- This compares numerical sampling behavior, not successful closed-loop rollout rates. Changing batch size also consumes RNG differently on subsequent calls, so seeded full rollouts need not follow identical sample sequences.
- AST/source comparison from May 7 to HEAD shows the only `PI05Pytorch.sample_actions` changes are validation/padding in the **guidance** branch, which is inactive here. KV-cache construction/expansion and denoising methods are unchanged. Existing `self.eval()` calls remain active.

Reproduction script and results: `code/compare_batch_sizes.py`, `code/batch_comparison.json`. Run from `/home/jeremiah/lerobot` with `/home/jeremiah/miniforge3/envs/openteach/bin/python`, GPU access, and sufficient free memory. The script uses saved observations/weights only.

## Normalization, state, and cameras

- Archived preprocessor/postprocessor JSON are semantically identical to checkpoint JSON. Every active state/action tensor is value- and shape-identical to the saved checkpoint; all other tensor values also match after flattening (some irrelevant count/scalar fields serialize rank 0 instead of rank 1).
- The checkpoint explicitly uses **STATE QUANTILES, ACTION MIN_MAX, VISUAL IDENTITY**. The class's default ACTION QUANTILES does not override the loaded checkpoint. March 10 `5f40a04c` introduced action MIN_MAX before checkpoint training. No processor/normalization implementation changes exist between the March baseline and HEAD on this branch.
- State ordering remains joint positions 1–7 then gripper width; action ordering remains XYZ, quaternion XYZW, gripper. Camera keys are preserved and `_preprocess_images` follows the checkpoint's front/side/wrist order. Relevant code: [observation construction](/home/jeremiah/lerobot/src/lerobot/robots/franka/franka.py:197), [model camera ordering](/home/jeremiah/lerobot/src/lerobot/policies/pi05/modelling_pi05_taco.py:1452).
- Across all seven runs, all three archived observation image tensors change at every chunk boundary: no consecutive duplicates. Raw state also changes with robot motion.
- For `dracorex_base_0` chunks **0, 3, 6, 10**, each archived side/wrist/front observation matched a frame in **cam_0/cam_1/cam_2**, respectively, with **exactly zero pixel difference** after the normal RGB conversion and front crop (columns 140:500). Thus the actual policy saw the changed scene observed in the videos; the streams were not swapped or frozen.
- Matching frame timestamps are **50–115 ms before the chunk's step-start timestamp**. This precedes ~443 ms policy generation. Camera timestamp synchronization imposes an interpretation limit on exact age, but there is no evidence of multi-second stale video.
- Front masking `[:, :140]=0`, `[:, 500:]=0`, BGR-to-RGB conversion, camera ports, image resizing, model camera ordering, and state-token construction are unchanged since March/May. Camera extrinsics changes in trajectory overlays would affect guidance visualization, not these unguided policy inputs.
- The data agent found starting joint angles and starting end-effector positions essentially unchanged between old/new rollouts. Extreme quantized state tokens therefore are not a newly introduced initial-state shift.

Camera evidence: `code/camera_identity.json`, `code/check_camera_identity.py`. Full provenance/timing audit: `code/recording_metadata.json`, `code/inspect_recording_metadata.py`.

## Recommended controlled checks

First reproduce the demonstrated pink/blue spatial arrangement while holding the current horizon and runtime fixed. Alternate original and reversed arrangements to isolate layout. Then compare 75 vs 100 and one-sample/no-synchronous-recording vs the current base sampler while holding checkpoint, reset pose, task, object positions, and camera calibration fixed. The investigation supports doing this before changing gains, normalization, quaternion conventions, or the checkpoint.

No tracked repository files were changed by this audit; all generated files are in the analysis artifact directory or `/tmp`.
