# OpenTeach regression audit

Read-only investigation of `/home/jeremiah/openteach` (`main`) and its imported Deoxys dependency. No robot connection, motion, camera connection, service restart, checkout, or source modification was performed. No `AGENTS.md` was found at the relevant ancestor/repository locations.

## Conclusion

The active LeRobot absolute Cartesian policy path has **no identified OpenTeach change that explains a large success-rate drop**. Many conspicuous April/May commits affect VR teleoperation or joint control, which this policy does not call. An offline comparison of March and current Cartesian controller arithmetic produced bit-identical actions on **12,752 recorded state/target pairs**, covering nine complete original-rollout files and seven current files. Current arithmetic also reproduces the logged arm commands exactly at the median and 95th percentile for every recording.

The largest remaining software uncertainty outside LeRobot is the **Deoxys C++ gripper worker changed May 18/22**, after the apparent May 9 date of original successful rollouts. Its Python arm and gripper command semantics are unchanged; installed/deployed C++ binary provenance on the robot controller is not established by these local repositories. Gripper command binarization upstream is worth checking because negative command magnitude directly controls open width.

## Scope and chronology

- OpenTeach current HEAD: `60a4701` (2026-05-25 21:30:15 -0400).
- Last March commit: `f1ec57c` (2026-03-12; playback delta-joint option).
- March 4 commit `16c18bb` restores ZMQ conflation and single-message lossless PNG transport, fixing stale camera frames. No later camera publisher/subscriber change is present.
- Recorded timestamps establish March 23 demonstrations, May 9 original rollouts, and September 11 (local time) recent rollouts. Treat **May 9 as the successful rollout baseline**. April changes therefore predate the recorded successful rollouts; exact runtime source provenance remains unrecorded.
- Working tree is dirty in `openteach/components/operators/franka.py`, `server/monitor.py`, and `visualize_demo.py`. Dirty Franka changes alter only VR translation/rotation scaling, VR viewpoint yaw (90 to 135 degrees), and controller-frame jump threshold (0.05 to 0.1 m). Monitor changes bind the web viewer publicly; visualization tolerates missing old torque fields. None is on the active Cartesian policy action path.
- No Git submodules are configured/reported.

## Active call path and exact arithmetic verification

Current LeRobot sets `DELTA_JOINT_ACTIONS = False`, `JOINT_ACTIONS = False` at `/home/jeremiah/lerobot/src/lerobot/robots/franka/franka.py:23`, constructs OpenTeach using `control_mode="absolute_eef_pose_to_delta"` at line 60, and calls `operator.arm_control(target_pose=abs_eef_pose, gripper_cmd=...)` at line 160.

Current OpenTeach:

- `/home/jeremiah/openteach/openteach/components/operators/franka.py:190` selects the velocity OSC config for `absolute_eef_pose_to_delta` (specific branch line 194).
- `franka.py:618` passes target pose to `get_abs_eef_pose_actions`.
- `franka.py:727` converts absolute target pose into Cartesian translation error and quaternion-relative axis-angle error; clips translation-vector norm to 0.1 and rotation-vector norm to 0.2; sends that delta to `OSC_POSE`.
- `franka.py:641` forwards action and controller configuration; `franka.py:647` forwards the supplied gripper command without teleop processing.
- `configs/osc-pose-controller-velocity.yml:3`: `is_delta: true`; lines 12–13: translation/rotation Kp 250; lines 16–17: scale 1; interpolation fraction 0.3 at line 7; state estimation false at line 22. File unchanged since March baseline.
- `franka.py:24`: 20 Hz control, 60 Hz state; unchanged.

Offline verification extracted the pure old `arm_control` arithmetic from `git show f1ec57c:openteach/components/operators/franka.py`, and the pure current `get_abs_eef_pose_actions` with Python AST. It evaluated both with recorded `cartesian_pose_cmd`, `eef_quat`, and `eef_pos`, using mock state objects and pure transform utilities. No interface was constructed.

Results in `/tmp/openteach_formula_comparison.json`:

- 12,752 recorded state/target samples, all nine complete original `.h5` command histories plus all seven recent histories.
- Maximum old-versus-current absolute difference: **0.0** for every recording.
- Recomputed-versus-recorded delta error median and 95th percentile: **0.0** for every recording.
- At 99th percentile, all were zero except recent run 1 (0.0003006); per-record maximum ranges 0.000659 to 0.001821. This is compatible with separate asynchronous state reads between computing and logging commands, and occurs in both periods; it is not a gross frame/units/sign conversion change.
- Original run 0 has a `.h5.<pid>.tmp` file and was not included in this complete-`.h5` arithmetic test.

## Major repository changes and relevance

| Commit/date | Change | Assessment for this policy |
|---|---|---|
| `16c18bb`, Mar 4 | Replaces raw multipart RGB with PNG single-message payloads and restores ZMQ_CONFLATE | Predates March 23 training recordings; no later reverse change. PNG is lossless. Not evidence of a recent regression. |
| `f520001` and `b0c811e`, Apr 13 | Adds GeoFIK joint control; decreases joint Kp from 200/150 to 75 and increases damping | Large dynamics difference **if joint impedance is selected**, but active policy selects OSC pose-to-delta and logged attrs reportedly agree. |
| `dbb0a1b`, Apr 15 | Adds explicit control modes; raises absolute OSC position config Kp 10/5 to 250/250 | Absolute-position OSC file is not used by this mode. Relative OSC file remains unchanged. |
| `4a33e8b`, Apr 15 | Adds direct joint playback, joint action low-pass filter (0.9 normal joint weight, 0.4 at joint index 1) | Filter only runs in `absolute_joint` (IK teleop), not `absolute_joint_direct`, and not active Cartesian mode; see `franka.py:613` and `:653`. |
| `12c35fa`, Apr 16 | Makes RGB recording codec configurable, defaults to lossless FFV1; changes recorded depth compression to LZF | File recording/shutdown change; not live observation contents or control geometry. |
| `088c955`, Apr 22; `8b102df`, Apr 24; `1ecbe09`, Apr 25 | VR translation/rotation scaling; filter reset on teleop reset | Bypassed by policy's direct target-pose call. |
| `860eb0a`, May 5; `356a61a`, May 6 | VR quaternion / controller-axis changes and partial revert | The alarming quaternion commit concerns VR pose mapping (`_get_remote_message`, `_controller_tracking`), not `get_abs_eef_pose_actions`; arithmetic check excludes active-policy math change. |
| `e03c77c`, May 8 | `automatic_gripper_reset=False` at interface initialization | Does affect initial gripper opening; could matter if replay starts closed without an explicit reset. Predates May 9 successful baseline if chronology confirmed. Current `franka.py:171`. |
| `d2bea10`, May 8 | Fixes VR gripper toggle initialization and missing recording commands | Supplied policy gripper commands bypass `_get_gripper_message`. |
| `474db67`, May 11 | Holds last IK joint action on no IK solution; prevents missing action logs | Affects joint IK path, not active Cartesian policy. |
| `108cd52`, May 22 | Exits on `done` above threshold | Only if `done` is passed into `arm_control`; current FrankaRobot call passes target_pose and gripper_cmd only. |
| `6abbd98`, May 25 | Rejects discontinuous VR controller-frame translations | Bypassed by direct policy action call. |

Inactive LeRobot joint-action flags have compatibility problems after the OpenTeach API change: `franka.py:149` and `:153` still use positional `arm_control(None,None,playback_actions=...)`, which current OpenTeach no longer accepts; replacing `velocity_controller_cfg` alone also no longer changes `controller_cfg`. These would produce a failure or incorrect mode if enabled; they do not explain the current successful execution of the Cartesian branch.

## Camera and physical-environment boundaries

The camera capture implementation and serial/order/resolution config did not change after March. `/home/jeremiah/openteach/configs/camera.yaml:7` lists three serials in stable order; width 640, height 360, 30 FPS, rotation zero at lines 13–17. `openteach/components/sensors/realsense.py:53` uses BGR8 and `:122` applies the configured rotation. The camera code does not explicitly fix RGB exposure, gain, or white balance, and does not record camera extrinsics. Physical relocation, mounts, lighting, camera firmware/device defaults, and per-start auto-exposure can therefore change without a Git diff. The local source alone cannot rule these out; image comparison is the stronger evidence.

Current camera viewer was updated May 24, but is a separate diagnostic display; centered crop boundaries and crosshairs do not modify images going into policy inference.

## Imported Deoxys dependency

The OpenTeach environment resolves `deoxys` to `/home/jeremiah/deoxys_control/deoxys/deoxys/__init__.py`. This adjacent repo is on `main`, HEAD `138755e` (2026-05-29), with one dirty reset-script tolerance edit.

Compared `b20e16f` (Feb 24, last pre-March change) versus current using AST normalized without line metadata:

- `FrankaInterface.control`: identical.
- `FrankaInterface.gripper_control`: identical.
- `last_eef_pose`, `last_eef_quat_and_pos`, `last_q`, `last_gripper_q`: identical.
- No arm controller C++ or transform utility commits since March.

The significant later changes are in gripper infrastructure:

- `f06ebca`, May 18, introduces Python stop/retry handling and C++ gripper adjustments.
- `a90289c`, May 22, rewrites exception/thread handling in `deoxys/franka-interface/src/gripper_control_node.cpp`, uses nonblocking receive plus 1 ms sleeps, changes shared-command locking, fixes homing message type, and handles aborted gripper operations. Current source lines 111–151 receive messages; lines 157–229 execute them. Such changes could alter transient responsiveness; deployment of this exact binary must be checked before assigning causality.
- `4dcaff7`, May 22, removes Python stop handling; commit explicitly says the real issue was gripper commands not being binarized properly. The **final Python gripper function is AST-identical to March**.
- `138755e`, May 29, suppresses startup prints only.

Current and historical gripper mapping at `/home/jeremiah/deoxys_control/deoxys/deoxys/franka_interface/franka_interface.py:676`:

- Negative action opens to `0.08 * abs(action)` m at 0.1 m/s. Therefore -0.25 commands 20 mm, -0.5 commands 40 mm, -1 commands 80 mm. Negative does **not** unconditionally mean fully open.
- Nonnegative action grasps, target width -0.01, speed 0.5, force 30 N, epsilon 0.08. See lines 686–692.

Thus changed upstream binarization or fractional policy outputs can produce a major change in physical grasp behavior even with identical low-level gripper source. Compare actual recorded `gripper_action` distributions and width transitions, and trace the active policy postprocessor before changing gripper implementation.

Reset-script changes (`846ed2e`, Mar 27; Apr 15/20 side/unplug presets; May 22 reset to file endpoint) create alternative initial conditions only when their flags are used. Current dirty `/home/jeremiah/deoxys_control/deoxys/examples/reset_robot_joints.py:166` changes reset acceptance from 0.001 to 0.005 rad. The default upright pose vector is unchanged. Use recorded initial joint/EEF pose to assess impact instead of assuming changed reset behavior.

## Appropriate next checks

1. Combine exact arm arithmetic equivalence with recorded controller attrs and action-to-motion analysis. If gains/config match and commands match, focus on policy observation/output differences and physical scene, not VR teleop changes.
2. Check recorded gripper open/close magnitudes and delays, and the upstream binarization path; this is an actual low-level semantic sharp edge with an explicitly documented history of failures.
3. Record exact installed Python paths/package versions and the robot-side Deoxys binary build/commit and robot model/tool transform/payload settings for a controlled future experiment. Local source is insufficient to establish firmware, runtime payload, calibration, or binary equality.
4. Any A/B replay should reproduce initial pose, gripper opening, scene layout, camera crop/order, policy prompt/checkpoint, and inference cadence. Do not attribute a change in whole-trajectory speed to plant dynamics without controlling commanded target magnitude and policy output.
