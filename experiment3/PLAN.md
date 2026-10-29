# Experiment 3: Egocentric pen demonstration to G1 dart pickup

Prepared 29 October 2026. Status: step 1 scene/reset audit and phase 2 WiLoR setup/30-frame pilot completed. See `PHASE2.md` and `ASTRA_HANDOFF.md`. Pickup retargeting and pickup training have **not** yet been run.

User update: HaMeR is now selected, with WiLoR preserved. HaMeR setup and 30-frame full-mesh pilot are complete (28 valid surfaces); see `HAMER.md`. The workflow below remains applicable with HaMeR as the hand front end. Full surface predictions do not validate hidden contacts or recover a calibrated robot/world trajectory.

## 1. Objective and recommended approach

Teach the G1 to reach for a dart resting on a table, acquire it with its **left Inspire DFQ hand**, lift it, and bring it into the established throwing configuration. Preserve the requested grip: index and middle together on one side of the barrel, thumb folded into opposition, ring and little fingers curled clear. This is the preparation skill that precedes Experiment 2.

The user has authorized changing the proposed workflow to achieve that outcome. Recommended pipeline:

**Personal video → reviewed hand and object reference → constrained robot retargeting → physically successful pickup controller → residual learning → continuous handoff to throwing.**

Use WiLoR as a candidate hand reconstruction front end. A video-derived pose sequence alone is insufficient to teach reliable pickup: it lacks robot actions, measured contact forces, reliable metric object motion, and robot-specific collision constraints. Begin with an interpretable controller and a physical feasibility test, then train corrections around successful executions. If WiLoR loses the grasp behind the fingers, retain its reliable reach and wrist information and fit the missing contact phase using the existing robot grasp and object geometry. Do not make reconstruction quality a prerequisite for every part of the task.

First scope: supported pelvis, fixed feet, one reachable dart, simulator-state feedback. Later scope: varied placements and integrated pickup-to-throw. Walking from a remote table and unsupported balance require an additional locomotion/manipulation stage. Egocentric **demonstration collection** does not imply an egocentric-camera-only deployed policy.

## 2. Evidence from the supplied video

Source: `C:\Users\Amarnath\Downloads\IMG_4238.mov`.

| Property | Inspected value |
|---|---|
| SHA-256 | `f94061379323b6315390d855680a3728c8eb29942da30ae024861152031f92cf` |
| Size | 1,730,895 bytes |
| Duration | 7.135 s |
| Video | H.264, 480 × 854 portrait, 421 reported frames |
| Encoded frame rates | nominal 59 FPS; average 84200/1427 ≈ 59.005 FPS |
| Timestamp time base | 1/600 s |
| Color tags | BT.2020 primaries, HLG transfer, 8-bit YUV |
| Decode check | FFmpeg full audio/video decode completed without reported errors |
| Visual inspection | Samples at approximately 0.5-second intervals; no framewise 3-D inference |

Inspection sheet: [video_contact_sheet.jpg](research/video_contact_sheet.jpg), read left-to-right then top-to-bottom. The final black tile is padding, not a source frame. Color appearance needs verification against a deliberately converted SDR preview before model inference; metadata can be inconsistent with previously transcoded pixels.

The clip shows one pen on a wooden table, an approaching hand, finger closure, apparent pickup, and a wrist turn that presents the pen and palm. Phase 2's denser timestamped review supersedes the original coarse windows: 0–1 s initial object, 1–2.25 s approach, 2.25–4.75 s acquisition, 4.75–5.75 s lift/rotation, then held presentation. These are sampling-based annotations, not verified contact or liftoff timestamps. The user confirmed **left hand, unmirrored**; the earlier right-hand visual guess was incorrect. Use left-to-left mapping, without reflecting the image.

The pen and contact surfaces become occluded during acquisition. There is no visible calibration marker, supplied pen measurement, depth stream, or camera calibration. The viewpoint changes, especially near the end. A single RGB view cannot establish precise metric lift height or all finger contacts. Preserve original presentation timestamps rather than assuming exactly 60 FPS. The final view helps review finger roles but is not a demonstrated robot throwing pose.

This is adequate for a first reconstruction and task-sequence pilot. Obtain pen length/diameter and camera information if available. If contact reconstruction is poor, request additional well-lit views or new recordings with a measured marker board; continue robot feasibility work meanwhile. For later generalization, collect roughly 20–30 pickups over several placements and recording sessions, including failures. This is a proposed collection target, not a guarantee of sufficiency. Hold out complete recordings/sessions; adjacent frames of this one clip cannot constitute an independent test set.

## 3. What the existing robot already provides

Inspected local sources: [`inspire_model.py`](../inspire_model.py), [`inspire_grip.py`](../inspire_grip.py), [`policy.json`](../inspire_grip/policy.json), [`grip_reference.json`](../inspire_grip/reference/grip_reference.json), [`left_hand_grip.png`](../inspire_grip/left_hand_grip.png), [`experiment2/env.py`](../experiment2/env.py), and [`experiment1_throw.py`](../experiment1_throw.py).

The hand has six independent motors and twelve moving joints with mechanical couplings. Optimize the independent motors and derive coupled joints; copying twenty-one human landmarks into twenty-one independent robot joints would be incorrect. Existing hand control caps command torque at 0.35 Nm. Reference rates are 0.5 rad/s except thumb pitch, which uses 0.5/2.4 rad/s to respect dependent thumb motion. Measured speed can still exceed reference speed under contact loads.

The saved grasp targets, in radians, are:

| Independent left joint | Grasp target |
|---|---:|
| Thumb proximal yaw | 1.242099 |
| Thumb proximal pitch | 0.370179 |
| Index proximal | 0.795016 |
| Middle proximal | 0.793756 |
| Ring proximal | 1.550000 |
| Pinky proximal | 1.550000 |

These are controller reference values, not mandatory exact measured angles. Feedback and contact change the settled configuration. The learned thumb/index/middle force targets are approximately 2.189/1.094/0.915 N. They are useful starting values for holding, not demonstrated optimal forces for acquisition.

The dart-body origin relative to `left_wrist_yaw_link` is `[0.189487, -0.056132, 0.023476]` m with local quaternion `[0.707107, 0, 0.707107, 0]` in **wxyz** order. The dart's local +X points toward its tip. Preserve this transform as a terminal grasp reference, distinguish body origin from COM, and validate it with the actual mesh/contact surfaces.

Existing reset places the dart directly in the hand. Pickup requires a new reset that places the free dart on the table and starts the hand clear of it. Reusing `InspireGrip.reset()` unchanged would silently bypass the objective. Existing open/closed finger logic also needs an experiment-local extension to command preshape, acquisition, and holding independently.

### Table placement and accessibility

`build_scene.py` places the table center at `[0.48, -1.20, 0.77]` m, with half-size `[0.48, 0.28, 0.025]`; the upper surface is z = 0.795 m. A tray top lies at z = 0.807 m. The first spare dart starts at `[0.46, -1.28, 0.832]` m and is free; the other spare darts are fixed props. The table is on the negative-Y/right side, while the active throwing hand is left.

Do a collision-aware reachability sweep before planning pickup from that location. Its accessibility from the supported pose has not been established. For the first curriculum, use a separate scene variant with a table on the left/front, chosen by reachable **grasp poses**, not just reachable wrist points. Keep the same room appearance. Return to the original table location only with a verified reach or a later base-repositioning controller. Interpret “other side of the room” as a separate experiment view/environment, not permission to assume the supported robot can walk there.

A thin dart lying flat can prevent the thumb from getting underneath. Test approach clearance using full fingers, barrel, point, shaft, flights, and tabletop. If necessary, begin with a barrel-access cradle or slightly overhanging placement, then remove that assistance through curriculum. Report assisted and flat-table results separately. A pen has different size, balance, and no flights; transfer finger roles and approach phases, then solve the dart grasp afresh.

## 4. Research findings and method selection

| Source | Relevant finding | Decision for this experiment |
|---|---|---|
| [WiLoR, CVPR 2025](https://arxiv.org/abs/2409.12259) | Combines hand localization/handedness with 3-D reconstruction; its video coherence does not depend on a temporal model. | Use it for hand shape and articulation estimates. Add explicit identity association, uncertainty review, and temporal filtering. |
| [Official WiLoR implementation](https://github.com/rolpotamias/WiLoR) | Provides model checkpoints and an image-folder demo; requires MANO assets. | Prefer this reproducible baseline and pin the revision. Treat WiLoR-mini as an optional demo convenience. |
| [WiLoR demo source](https://raw.githubusercontent.com/rolpotamias/WiLoR/main/demo.py) | Exposes 3-D joints, vertices, handedness, and camera translation computed with an assumed focal model. | Build an exporter; validate units and coordinate conventions rather than treating the render as calibrated world motion. |
| [Dex Retargeting](https://github.com/dexsuite/dex-retargeting) | Supplies optimization tools for mapping human hand motion into robot configurations. | Evaluate vector/contact retargeting against a small custom optimizer using the existing MuJoCo model. Audit any Inspire configuration against this exact DFQ asset. |
| [DexMV, ECCV 2022](https://yzqin.github.io/dexmv/) | Translates reconstructed hand **and object** motion into demonstrations for robot learning. | Include the object and validate physical execution before using demonstrations for imitation. |
| [EgoAERO, June 2026 preprint](https://arxiv.org/abs/2606.08057) | Describes egocentric RGB-D hand-object reconstruction, camera-motion compensation, contact optimization, and residual learning. | Useful architectural precedent, but its depth input differs from this RGB MOV. Do not assume equivalent reconstruction or performance. |
| [OpenCV marker-pose documentation](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html) | Known marker geometry and camera calibration support camera pose estimation. | Use a measured marker board in future recordings to anchor a table frame. |
| [MuJoCo contact documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html) | Frictional contact is modeled through approximate soft constraints. | Validate grasp robustness and timestep sensitivity; successful simulation alone does not establish hardware transfer. |

These sources support components of the design. None establishes that this particular clip and G1 hand will produce a successful pickup. The curriculum, controller choices, thresholds, and comparisons below are project proposals.

## 5. Reconstruction and coordinate handling

1. Save source hash, ffprobe metadata, and per-frame timestamps. Produce an inspected upright RGB/SDR inference copy with an explicit source-frame map. Retain the original unchanged.
2. Run a short WiLoR pilot spanning open hand, closure, occlusion, and final presentation. Verify handedness, fingertip placement, and wrist orientation in overlays before processing every frame.
3. Export raw joints/vertices, predicted camera quantities, hand identity, box confidence, and all transformations. Detection confidence is not per-joint depth certainty. Add separate visibility and reconstruction-quality flags.
4. Track the pen separately: annotate endpoints, barrel axis, grasp region, and visibility. For this short pilot, manual keyframes plus tracking and human correction are preferable to an unvalidated automatic 6-D object tracker. The pen's axial rotation may be ambiguous; do not invent it.
5. Use hand-relative and object-relative features where metric camera motion is unknown. Before contact, visible table/object landmarks constrain the approach; during occlusion, mark uncertainty. Hand-object co-motion can support a grasp hypothesis, but does not measure contact force.
6. Smooth within reliable segments and preserve original timing. Interpolate only short, reviewed gaps. Record long gaps instead of manufacturing ground-truth trajectories through them.

Use named frames: camera C, table/world W, human hand H, object O, and robot base B. With the convention that `T_A_B` maps B coordinates into A, a calibrated hand pose obeys `T_W_H(t) = T_W_C(t) T_C_H(t)`. Egocentric camera motion makes `T_W_C` time-dependent. A planar homography can stabilize table points but cannot recover arbitrary 3-D hand motion above the plane.

For the current uncalibrated clip, use normalized hand articulation, object-axis relationships, and phase order as the primary demonstration. Fit absolute robot wrist motion to the known simulated table/dart. Label that trajectory as robot-planned motion guided by video. If a calibrated recording becomes available, compare metric retargeting against this baseline.

If right-to-left conversion is required, reflect landmark geometry in a defined anatomical frame and reconstruct an orthonormal left-hand frame. A reflection has determinant −1 and cannot be passed directly as a rotation quaternion. Check thumb opposition and fingertip ordering visually on the actual left model. Human joint angles should not simply be negated.

## 6. Retargeting, grasp feasibility, and control

Represent the demonstration by approach direction, palm orientation, thumb-to-index/middle vectors, closure sequence, barrel-relative contact locations, lift direction, and wrist turn. Weight visible thumb/index/middle features most strongly; use the established tucked pose for ring/little fingers where the demonstration conflicts with dart clearance.

Solve hand retargeting with joint bounds, mimic couplings, smoothness, collision penalties, and a terminal grasp prior. A proposed objective is:

`L = landmark-vector error + barrel-contact error + temporal change cost + joint-limit/collision penalties + terminal-grasp error`.

Use robust losses and visibility-dependent weights. Solve the seven-joint left arm with collision-aware IK and a smooth path, jointly checking the hand geometry. Retargeting produces references; replaying joint positions kinematically is only a diagnostic. Physical validation must execute bounded motor commands.

The first controller should use these contact-conditioned phases:

| Phase | Controller behavior and exit condition |
|---|---|
| Reach | Move an open/preshaped hand above and beside the barrel; remain clear of the table and flights. |
| Align | Position thumb opposite index/middle with an approach that admits all contact surfaces. |
| Acquire | Close gradually; regulate force after contact. Require sustained intended contacts rather than elapsed time alone. |
| Lift | Raise slowly while monitoring slip and dart-table separation. Return to acquisition or terminate on loss of grasp. |
| Transport | Keep the contact grasp while moving toward the established inward-wrist hold pose. |
| Prepare | Move from the hold pose into Experiment 2's backswing/start configuration, settle, then pass control. |

Use the same free-body dart, mass/inertia, contact geometry, and custom air loads. Object placement is allowed at episode reset only. Do not weld it to the hand, prescribe its trajectory, or add lifting forces to make a failed grasp appear successful. Diagnostic pregrasp resets may train later stages, but the final metric always starts with the dart resting on the table.

## 7. Learning strategy and experimental comparisons

First establish one physically feasible acquisition with trajectory/closure optimization. Search a small set of approach offsets, palm orientation, closure timing, and force targets. The established hold controller reduces the search space but does not prove a collision-free path into that grasp.

Next collect successful **physics-executed** controller trajectories. Behavior cloning can initialize a policy from their observations/actions; WiLoR landmarks alone are not robot action labels. Train residual PPO around the reference/controller, retaining the repository's SB3 infrastructure where useful.

Initial action proposal: residual targets for seven left-arm joints and six independent hand motors; keep waist fixed until reachability evidence justifies adding waist yaw/pitch. Use bounded target rates, existing actuator ceilings, and stable torque tracking. Begin at 50 Hz control and 0.5 ms physics to match existing infrastructure, then benchmark. Hand acquisition needs continuous commands; Experiment 2's single latched opening action is unsuitable for pickup.

Observations should include named joint positions/velocities, independent hand state, dart pose/velocity relative to hand and table, intended-contact forces, distances to table/goal, phase/reference, and previous action. Mark simulator-only measurements explicitly. Vision-only deployment is a later experiment requiring perception noise/latency and occlusion handling.

Reward progress through bounded or one-time milestones: approach, correct opposed barrel contacts, lift with maintained grasp, arrival, and settled readiness. Add costs for slip, force spikes, table penetration, flight/shaft grasp, joint violations, and abrupt actions. Mere object height is insufficient: it can reward batting the dart upward. Require sustained hand contact, low relative motion, and positive tabletop clearance. Avoid indefinite contact or waiting bonuses.

Curriculum: reachable assisted grasp → flat-table acquisition → lift → transport → start-pose handoff → randomized complete sequence. Start with fixed placement, then add position/yaw variation, friction/mass uncertainty, action latency, and reference perturbation. Record every distribution and distinguish curriculum training resets from full-task evaluation.

Compare under matched physics-step budgets:

1. Hand-authored controller/trajectory search without video.
2. WiLoR-guided reference with physical tracking but no learned residual.
3. WiLoR-guided reference plus residual PPO.
4. Reference-free PPO with equal actuator authority, if compute permits.

Compare hand-only versus object-aware references to test whether the added reconstruction work helps. A single-clip success is a feasibility result; multiple seeds and held-out placements are required to claim robustness.

## 8. Handoff to Experiment 2: the key integration contract

The saved static grip pose is **not** Experiment 2's initial arm pose. `make_simple_plan()` sets backswing shoulder pitch/elbow/wrist pitch to approximately `[-0.99, -0.48, -0.60]` rad while retaining other saved arm values. `DartThrowEnv` also initializes waist yaw/pitch to `[-0.025, -0.01]`, sets the unused right hand, and settles for 0.25 seconds.

Therefore use two targets: first reach the familiar static hold configuration; then physically move to the actual throw-start configuration. Export a versioned target contract from the pinned throwing environment, including settled measured joint state, hand-relative dart transform, acceptable velocities/contact forces, and controller state. Capture the settled distribution, not just XML or JSON targets.

During final integration, preserve the live MuJoCo object/robot state and controller continuity. Do **not** call the current throwing `reset()`, which replaces the acquired dart in the hand. Provide a separate adapter that initializes episode bookkeeping and policy phase from the measured handoff state. Preserve or deliberately blend finger command/force-integrator state and arm targets; explicitly handle residual history, drift reference, phase time, and release flags. Observation normalization and the exact policy checkpoint must be paired and frozen for evaluation.

Matching the visible pose is insufficient: finger forces, dart transform, velocities, and the policy's expected initial-state distribution matter. First test holding after transfer of control, then throws. If pickup endpoints fall outside the throwing policy's tolerance, fine-tune a **copy** of the throwing policy on those endpoints. Keep standalone pickup success, readiness success, and end-to-end throw outcomes as separate measurements. Current throwing accuracy cannot be assumed solved.

## 9. Validation and proposed acceptance gates

All numbers below are initial engineering targets to review after feasibility measurements, not current results or hardware specifications.

| Gate | Evidence required |
|---|---|
| Reference | Timestamped overlays; confirmed hand mapping; visible contact phases reviewed; uncertain intervals explicit. |
| Kinematics | Reachable approach and destination, valid coupled joints, clearance checked with full hand/dart/table geometry. |
| Pickup | From tabletop, dart clears table by at least 30 mm at its lowest collision point and stays supported by intended fingers for 1 s. |
| Grip | Thumb/index/middle barrel contact; ring/little clear; relative translation drift below 2 mm and rotation drift below 5° over the 1 s hold. |
| Readiness | Dart relative pose within initial 5 mm/5° tolerance of the measured target; arm/waist error below 0.05 rad; dart speed below 0.05 m/s and angular speed below 0.2 rad/s for 0.5 s. |
| Integration | Continuous physical pickup → preparation → throwing control with no grasp reset, state teleport, or artificial object support. |
| Numerical | Stable at original and half timestep; report penetration, force, measured joint speed, and joint-limit errors. |

The readiness tolerances must be tightened or relaxed based on measured downstream policy tolerance; passing them alone does not certify compatibility. Use contact geometry extrema for clearance, not COM height alone. Calibrate penetration tolerances against the existing soft-contact model and log magnitudes even below the failure threshold.

Pilot evaluation: 20 fixed-setup complete attempts. Robustness evaluation: at least 100 randomized full-start attempts per training seed, preferably three seeds, with explicit held-out position/yaw ranges. Count failed initialization, slips, drops, wrong-finger grasps, and collisions in the denominator. Report pickup rate, readiness rate, completion time, slip/drift, forces, violations, and failure types, with binomial confidence intervals. Save all outcomes and states; render representative failures and a fixed evaluation set. Report assisted-cradle and flat-table performance separately.

## 10. Isolation, dependencies, and deliverables

Implement under `hackathon/experiment3/`, using a scene/controller variant and separate outputs. Existing throwing workers own independent simulations; editing shared files can affect new workers or later resumes even if current workers have imported code already. Freeze source hashes and copy required configurations/assets into experiment snapshots. Add an integration adapter separately once pickup works. Do not modify the active throwing environment or normalization files during development.

The inspected GPU is an RTX 3070 with 8 GiB VRAM. One inspection showed about 2.1 GiB used while Experiment 2 reported CUDA policy/optimizer state and four CPU physics workers. This is a transient snapshot, not a reservation. Use a separate Python environment for WiLoR, start with a few frames and batch size one, and measure peak VRAM/latency. Run CPU preparation/retargeting alongside throwing; queue GPU-heavy inference/training if contention occurs. “Parallel experiment” does not require competing long GPU jobs. WiLoR's documented PyTorch 2.0/CUDA 11.7 setup should not overwrite the working PPO environment. Native Windows support must be tested; WSL is a fallback, not an assumed prerequisite.

The official instructions require `MANO_RIGHT.pkl` through MANO registration. Its availability has not been established across this machine. WiLoR model terms are identified as [CC BY-NC-ND 4.0](https://raw.githubusercontent.com/rolpotamias/WiLoR/main/license.txt); MANO and Ultralytics have separate terms. Record asset provenance and keep model/data licenses distinct from project code, particularly for a later public hackathon release. No checkpoint downloads or dependency installation were performed for this plan.

Proposed artifacts:

```text
experiment3/
  PLAN.md                         # this researched plan
  research/video_contact_sheet.jpg
  data/source_manifest.json       # source hash, media metadata, timing
  data/annotations.json           # phases, object axis, visibility
  data/hand_raw.npz               # WiLoR output with conventions
  data/reference.npz              # filtered, reviewed reference
  configs/scene.json              # table/dart setup and curriculum
  configs/grasp_target.json       # pinned existing contact grasp
  configs/handoff_target.json     # measured throw-start contract
  prepare_video.py
  reconstruct_hand.py
  retarget.py
  env.py
  train.py
  evaluate.py
  integrate_throw.py
  runs/<run_id>/                  # sources, seeds, states, metrics, videos
```

Step 1 now has an isolated scene builder, empty-hand reset, and kinematic/clearance audit; see `README.md` and `research/setup_audit.json` for the current front-table setup and measured results. Later pipeline artifacts in the proposed layout remain unimplemented. Personal media and generated model files stay local via experiment-specific ignore rules. Pin revisions/checkpoint hashes and record coordinate, joint-name, normalization, and action-schema versions.

## 11. Implementation order

1. Build a reachable experiment-only table setup and a correct empty-hand/tabletop-dart reset. Audit flat-table fingertip clearance immediately.
2. In parallel as independent work, prepare timestamped video annotations and a small WiLoR overlay pilot. Proceed with robot grasp feasibility even if MANO acquisition or reconstruction needs attention.
3. Fit a collision-aware reach/acquire/lift trajectory guided by visible video phases and the existing terminal grasp. Execute physically and save failures.
4. Establish repeated pickup and transport using the baseline controller before spending a long training budget.
5. Train and evaluate residual corrections, with held-out placements and comparisons against the no-video baseline.
6. Export the pinned throwing-start contract; validate continuous handoff and then end-to-end behavior.

The first demonstrable deliverable should be a video showing the empty robot hand approach a stationary tabletop dart, establish the specified grip, lift it, and hold it in the existing configuration, with contact and drift measurements beside the replay. That directly tests the main uncertainty before expanding to learning and integration.
