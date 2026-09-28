# Recorded dynamics and timing comparison

The strongest robot-log evidence is a change in **commanded pickup location**, rather than a failure of the robot to reach its commanded target. Every original run changes robot-y side between its first and second sustained grasp attempts; no recent run does. This matches the visual investigation's finding that block colors were swapped between the old and new layouts. The old/new logs otherwise show comparable initial posture, Cartesian tracking, and gripper closing response. Longer and more frequent inference pauses are a separate, real software/runtime change.

## Data and chronology

| Group | Runs | Robot samples | Internal timestamp dates (UTC) |
|---|---:|---:|---|
| Training demonstrations | 50 | 27,481 | March 23, 2026, 19:27–20:39 |
| `base_rollouts_original` | 10 | 6,727 | **May 9, 2026**, 16:58–17:13 |
| `demonstration_dracorex_base_*` | 7 | 6,825 | September 12, 2026, 00:38–01:38 (September 11 locally) |

Original run 0 is readable HDF5 despite the suffix `.h5.884292.tmp`. Its controller attributes were never written; other original runs carry them. All analyses include this tenth run. Processed demo HDF5 attributes such as `openteach current commit` may describe extraction provenance and must not be assumed to identify collection code.

No logs have reliable task-success labels: original logs lack `done`; all recent values are `b'null'`. Gripper measurements below are grasp proxies, not task success rates. The original folder itself contains failed grasps and cannot be treated as ten perfectly successful episodes.

## Pickup-site behavior

Only closed-command segments lasting at least two seconds count as sustained grasp attempts. At the onset of each segment, compare `cartesian_pose_cmd[:,1]` (commanded y) and `eef_pos[:,1]` (measured y). Positive/negative robot y refer to fixed sides of the workspace. Avoid assigning colors from numeric logs alone; the visual investigation establishes colors separately.

| Recent run | First target y | Second sustained target y | Third sustained target y | Sustained object captures |
|---|---:|---:|---:|---:|
| 0 | +54 mm | +53 mm | +53 mm | 1 |
| 1 | +55 mm | +57 mm | +51 mm | 1 |
| 2 | −107 mm | −37 mm | −38 mm | 1 |
| 3 | +62 mm | +54 mm | +54 mm | 1 |
| 4 | −87 mm | −96 mm | −74 mm | 1 (on second attempt) |
| 5 | +61 mm | +52 mm | +59 mm | 1 |
| 6 | +64 mm | +58 mm | +57 mm | 1 |

Nine original runs first target positive y, then target negative y. Original run 8 starts negative and then targets positive. **All ten switch sides. All seven recent runs stay on their original side**, with repeated empty grasps after one object capture. In recent positive-y runs, later target y values remain +51 to +59 mm while the remaining pickup site is around −100 mm. The measured position follows these misguided targets; a mechanical tracking defect cannot explain this roughly 15 cm error in desired location.

See `grasp_site_comparison.png` and `sustained_grasp_sites.csv`. Original episodes with failed second grasps still switch to the appropriate side: this separates their misses from the recent repeated wrong-site behavior.

## Timing changes

| Statistic | Demos | Original | Recent |
|---|---:|---:|---:|
| All-command median interval | 50.04 ms | 49.90 ms | 50.30 ms |
| All-command p99 interval | 50.76 ms | 346.38 ms | 492.53 ms |
| Within-chunk median interval (exclude startup) | — | 49.84 ms | 50.22 ms |
| Inter-chunk median interval | — | **378.66 ms** | **499.48 ms** |
| Inter-chunk interval range | — | 339.46–405.05 ms | 472.60–614.38 ms |
| Execution horizon established by gap periodicity | — | **100 commands** | **75 commands** |

Original command gaps are precisely at multiples of 100; recent gaps are at multiples of 75. All recent runs have 975 commands (13 × 75). The median inter-chunk gap increased by 120.8 ms, about 32%. Combined with the shortened horizon, inference pauses occur more often: roughly every 3.75 seconds of action instead of every five seconds. There is also a separate approximately one-second startup interval in every group, which should not be mistaken for an inference pause. Recent trajectory metadata independently records a median 443 ms generation time and 75-action execution horizons (see the code audit).

The change can affect trajectories, but it does not by itself explain why requested later grasp targets stay on the wrong side. A controlled old-layout and old-horizon replay is needed to isolate its contribution.

## Dynamics evidence

| Pooled statistic | Demos | Original | Recent |
|---|---:|---:|---:|
| Median simultaneous command–state position difference | 11.44 mm | 10.86 mm | 10.20 mm |
| p95 simultaneous command–state position difference | 21.05 mm | 22.01 mm | 22.51 mm |
| Median target vs next recorded state position difference | 9.59 mm | 9.15 mm | 8.45 mm |
| Median orientation command–state difference | 0.0517 rad | 0.0489 rad | 0.0468 rad |
| Median measured speed during normal intervals | 45.7 mm/s | 40.9 mm/s | 39.4 mm/s |
| Median close-command to width below 60 mm | 0.701 s | 0.698 s | 0.679 s |
| Sustained object-width / all sustained closed segments | 100/100 | 17/25 | 7/21 |

Position/orientation differences use the observation immediately before sending its corresponding command. They are ongoing trajectory errors, **not** final settled target accuracy. Normal intervals are 25–80 ms; next-state comparisons omit inference/startup gaps. The populations follow different trajectories and encounter different contacts, so these descriptive numbers cannot prove identical intrinsic robot dynamics.

Nonetheless, the available measurements argue against a gross deterioration:

- Initial mean xyz is `[0.45761339, 0.03196682, 0.26526201]` m original and `[0.45762160, 0.03197765, 0.26524599]` m recent, differing by about 21 micrometers overall. Median initial joints agree within roughly 0.00005 rad. Both are inside the training demonstrations' initial posture range.
- `last_F_T_EE` and `last_F_T_NE` are exactly identical in all original and recent samples: the same ±45-degree planar rotation and 0.1034 m tool offset.
- Controller metadata agrees for every original file that has attributes and all recent files: OSC_POSE, `absolute_eef_pose_to_delta`, 20 Hz control, 60 Hz state, translation/rotation limits 0.1/0.2, Kp=250 for translation and rotation, unit action scaling, identical residual mass vector, identical state-estimator settings, and linear interpolation `time_fraction=0.3`.
- No recorded translation or rotation command hits its configured clipping norm in any of the three groups.
- Six of seven recent runs capture an object on the first sustained attempt. Run 4 misses first, then captures on its second attempt. Sustained captures plateau near 49 mm in all groups.
- Empty closed fingers are about 0.18 mm recent versus 0.28 mm original; open width is about 80 mm in both. This tiny change is far smaller than object size and does not explain the commanded wrong-side grasps.
- Original and recent policies both issue fractional gripper commands, including values below −1. The per-run fraction below −1 is 27.8–42.3% original versus 39.6–46.2% recent. Gripper closing delay has not increased.

Two late recent closed segments have apparent median widths around 33/38 mm because recording stops while closure is still developing. They are shorter than two seconds and are excluded from sustained capture counts. `grasp_events.csv` retains all events with their durations.

`last_eef_pose_d` is mostly a static reset pose in original runs and some recent runs; in other recent runs it jumps. In OSC torque control this field is not the commanded Cartesian target, and interpreting its difference as tracking failure would be incorrect. Controller/firmware binaries and low-rate logged states cannot establish the cause of this field's variation.

## Camera timing and log coverage

RGB metadata timestamps are Unix milliseconds; robot timestamps are Unix seconds. Videos start before control begins. Align by these timestamps, rather than by equal frame numbers or the video's nominal frame rate.

- Original cameras start about 12.9 s before the first robot command; recent cameras start about 1.2 s before it. Demo lead-ins vary about 2–16 s.
- Actual camera recording frequencies remain comparable: cam 0 median 26.45/26.33/26.86 Hz for demos/original/recent, cam 1 27.25/26.99/27.42 Hz, cam 2 25.33/25.32/26.11 Hz. This is below the nominal 30 Hz but not a new deterioration.
- Original run 4 continues recording video approximately **41.0 seconds after its last robot-log entry**; runs 7 and 9 continue about 13.8 and 23.5 seconds after theirs. Original run 4 has a 97.6-second camera recording but only 43.7 seconds of logged robot commands. Behavior in these tails cannot be paired with missing robot commands.
- Other original cameras end within about −0.13 to +0.29 s of the robot logs; all recent cameras end about 0.03–0.12 s afterward. Camera timestamps may precede host arrival by tens of milliseconds, so tiny negative end offsets are not evidence of a missing episode.

## Reproduction and outputs

From the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl-policy-regression /home/jeremiah/miniforge3/envs/openteach/bin/python analysis/policy_regression_20260912/dynamics/analyze_dynamics.py
```

The script reads user data and writes only this analysis directory. Required packages are already present in the OpenTeach Python environment: h5py, NumPy, SciPy, and Matplotlib. Camera metadata uses an unpickler that rejects all global/class construction.

- `runs.csv`: per-run timing, tracking, speed, clipping, gripper, and telemetry diagnostics.
- `aggregate.json`: pooled descriptive statistics and grasp-proxy counts.
- `grasp_events.csv`: every continuous closed-command segment.
- `sustained_grasp_sites.csv`: each run's sustained target/measured grasp locations.
- `command_gaps.csv`: every interval over 100 ms and its index modulo each horizon.
- `camera_timing.csv`: all 201 camera streams, frequency, frame intervals, and coverage offsets.
- `controller_metadata.json`: source HDF5 attributes, with original run 0 attributes absent.
- `grasp_site_comparison.png`, `group_comparison.png`, `original_tracking.png`, `recent_tracking.png`: static figures for reviewing behavior.

No robot commands, source changes, dataset changes, or external service calls were made.
