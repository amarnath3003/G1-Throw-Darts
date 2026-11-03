# HaMeR-guided G1 arm rehearsal — 2026-10-31

The existing left-hand video is sufficient for this arm-motion experiment. `retarget_arm.py` now solves the seven G1 shoulder/elbow/wrist joints together and executes the resulting reference using the existing torque controller in MuJoCo. The hand remains in the existing open preshape. This is a reach, hover and hand-lift rehearsal, **not a demonstrated dart pickup**.

## Approach

- Use the 28 valid HaMeR pilot frames. The saved joints are already left-handed; do not mirror them again. Build a palm frame from wrist, middle MCP, index MCP and pinky MCP. The anatomical-to-robot basis is a proper rotation with determinant +1.
- Anchor relative palm rotations to the approach interval, smooth them, cap the raw relative angle at 65 degrees and apply weights of 0.20 during occlusion and 0.45 elsewhere. These are intentionally softened orientation cues. Camera motion remains mixed into the source; the result is not an exact reproduction of the human wrist pose.
- Design the translation relative to the physically settled tabletop dart. Do not use HaMeR's assumed-focal-length camera translation as a metric world path. The reference stops above contact, then raises the empty hand.
- Solve bounded nonlinear least squares on the actual compiled G1 model. Optimize a point between the index/middle intermediate links, palm orientation, wrist/posture preferences, a low elbow preference and continuity. Preserve at least 0.10 radian joint-limit margin. The elbow is inferred by the solver, not observed in the video.
- Penalize exact geometry extrema below a conservative tabletop plane. Validate the complete interpolated trajectory against the finite table and torso geometry before execution. No robot collision masks are changed.
- Transition smoothly from the existing reset over four seconds; play the usable source interval at half speed. Total commanded motion: 16.27 seconds, followed by one second of settling.
- Execute through actuator torque targets. During the rollout, never assign robot or dart positions, weld the dart, or reset into a grasp. Replaying saved states is used only for rendering the recorded physics result.

This uses MuJoCo forward kinematics and collision queries with SciPy bounded least squares. An analytic closed-form G1 solver is unnecessary for this constrained trial. See the [MuJoCo kinematics/API documentation](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) and [official HaMeR implementation](https://github.com/geopavlakos/hamer).

## Validation and outputs

Run from `hackathon`:

```powershell
python experiment3/retarget_arm.py --render --validate-timesteps
python experiment3/check_arm_ik.py
```

Outputs are under `research/arm_ik/`:

- `report.json`: source/model hashes, conventions, per-reference IK errors, clearance, speed and physics checks.
- `trajectory.npz`: arm commands, time bases, target points/rotations and the local tool point.
- `rollout_0.0005.npz` / `rollout_0.00025.npz`: recorded physical state trajectories.
- `g1_arm_ik_rehearsal.mp4`: two camera views of the 0.5 ms physics rollout, explicitly labeled empty-hand rehearsal.
- `preview_*.jpg`: snapshots from that rollout.

The trial passed at both 0.5 ms and 0.25 ms. Planned maximum position/orientation residuals were 0.344 mm / 0.891 degrees. Minimum sampled dynamic environment clearance was about 17.5 mm. Maximum joint tracking error was about 0.049 radian (2.8 degrees); maximum sampled tool-point tracking error was 11.84 mm. That dynamic error needs attention before precision contact acquisition. The dart remained stationary. These errors describe the adjusted robot reference, **not accuracy against the actual human motion**. See the JSON for the final exact measurements and check frequencies.

`check_arm_ik.py` also verifies that a known 2.78 mm wrist/table penetration remains rejected by the clearance fallback, that all palm frames are proper rotations, that source/model hashes match, and that neither recorded rollout contains an artificial dart lift.

Contact checks inspect each physics step for new arm penetrations deeper than 0.2 mm. Positive separation is sampled at 30 Hz; kinematic clearance is checked every 20 ms. These are simulation checks, not a continuous-time or hardware guarantee. The pelvis remains supported, as in the existing experiment.

## Implementation findings

MuJoCo 3.10 sometimes returned zero distance for separated mesh/box pairs in this scene. The checker combines its native query with exact world-AABB separation, a conservative distance lower bound. This does not suppress contact or convert overlapping AABBs into clearance. The optimizer uses exact vertical mesh/primitive extrema, which are more stable under differentiation.

The pinned scene's XML roundtrip rounded dart inertia to six significant digits. The aerodynamic model correctly rejected that difference under its strict tolerance. The runner verifies the pinned code/profile hashes, bounds the discrepancy to under 5 ppm, verifies the inertia frame, and restores the canonical inertia **in memory only** (actual correction about 1.915 ppm). The original snapshot and source experiment remain unchanged. Aero forces are enabled during the rollout.

## Next step

Use this arm reference as the initial approach for a separate contact-aware acquisition controller. Fit the Inspire thumb/index/middle closure and a collision-feasible acquisition orientation, establish physical contact and load support, then lift the actual dart. Only after a sustained hold should it transition to the saved throwing grip. The existing closed throwing grip is still unsuitable directly on the flat table. No pickup policy has been trained, and Experiment 2 is untouched.
