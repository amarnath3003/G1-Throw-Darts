# Left five-finger dart grip

This milestone uses the **left Inspire DFQ five-finger hand** on G1. The thumb
opposes the index and middle fingertips on the dart barrel. The ring and little
fingers curl into the palm and must stay clear of the dart and flights.

The three photographs supplied by Amarnath define that contact topology. They are
manually interpreted references, not a calibrated 3-D reconstruction or a trained
image encoder. `inspire_grip/reference/grip_reference.json` records their role and
SHA-256 hashes; the original photographs remain local and are Git-ignored.

## Run

From this directory, with the project's Python environment:

```powershell
..\video_to_g1\.venv\Scripts\python.exe inspire_grip.py
..\video_to_g1\.venv\Scripts\python.exe inspire_grip.py --check
..\video_to_g1\.venv\Scripts\python.exe inspire_grip.py --render
..\video_to_g1\.venv\Scripts\python.exe inspire_grip.py --train
```

For a standalone checkout, create a Python 3.10+ environment and install
`requirements.txt`. The interactive controls need Tk; video export needs FFmpeg
on PATH. The checked-in `inspire_grip/policy.json` is sufficient to run the demo.

Controls: **O** opens the hand, **C** closes it, **R** reloads the dart into the
starting grasp, **Space** pauses, **1/2/3** select hand/opposite/robot views.
Closing an empty hand does not retrieve a fallen dart; use Reload for a new trial.
The viewer reports measured simulation contact normals, object height, and drift.

## Hardware model and physics

The hand comes from Unitree's published
[G1 Inspire DFQ descriptions](https://github.com/unitreerobotics/unitree_ros/tree/deeb607f3a8945a394f53068c28b3a88dde7ff06/robots/g1_description).
The unmodified URDFs, meshes, BSD license, and provenance are in `assets/inspire/`.
Both hands are articulated, and the active arm and controller are left-handed.
The left hand is instantiated from its own URDF, rather than reflecting a mesh:
its thumb mount differs slightly from the right hand and requires a separate fit.

`inspire_model.py` converts the link transforms, masses, full inertia tensors,
mesh collisions, axes, and ranges to MuJoCo. It preserves all twelve moving
joints per hand and the **six actuators** with mechanical mimic couplings. The
thumb pitch command is restricted to 0–0.5 rad because the source distal limits
and 1.6/2.4 coupling ratios require that range. The existing G1 body is retained;
the wrist inertial is replaced by the matching bare-wrist source inertial so the
old rigid rubber hand mass is not counted twice.

The dart retains its existing 26.7 g profile, COM, inertia, grooved barrel, steel
point, nylon stem, flights, and aerodynamic model. It has a free joint throughout
the rollout. Only reset positions it in the hand. No weld, mocap body, hidden
support, dart pose overwrite, or upward compensating force holds it. The twelve
equality constraints are finger-joint couplings only.

The pelvis remains supported by the visible fixture. Arm motors hold the pose
using torque-limited feedback. Hand commands are capped at 0.35 Nm, below the
URDF's 1 Nm limits. Reference motion is slew-limited to the URDF's 0.5 rad/s,
including the thumb's dependent joints. Actual joint velocity is reported in
validation; a slew limit is not a hard physical velocity clamp.

Contact friction is an **assumption**, nominally 0.7, tested from 0.55 to 0.9.
Contact softness, damping, armature, actuator dynamics, and idealized mimic
constraints have not been identified on hardware. Mesh collisions use MuJoCo's
convex representation. The elliptic friction solver uses 20 no-slip iterations
to reduce numerical drift; this is not an attachment constraint.

## What was learned

The contact pose is fitted within the hand's kinematics using the photo-derived
finger roles and actual collision surfaces. The reusable starting files are
`left_seed.json`, `pose_fit.json`, and `left_candidate.json`. `--fit` reruns the
geometric search; it does not overwrite the final learned policy.

`train_inspire_grip.py` performs seeded **cross-entropy policy search** over seven
parameters: three fingertip normal-force targets, force feedback gain, initial
closure offset, and two joint servo gains. It ran six generations of twelve
candidates, each evaluated in two randomized five-second hold/release episodes:
72 candidate evaluations / 144 episodes. The controller's observations are joint
position/velocity and simulated fingertip contact forces. The objective rewards
low drift, sustained three-finger contact, modest forces, and a clean release;
it penalizes contact with other fingers or non-barrel dart regions.

This is a trained low-dimensional force controller, **not residual PPO**, a VLA,
or a general grasp-from-camera policy. It starts with the dart in a fitted grasp.
Autonomous pickup, throwing, reacquisition after a miss, and whole-body balancing
remain separate tasks. Source parameters and every training evaluation are
saved in `training.json`; there is no opaque neural checkpoint.

## Evidence

After training, the user requested an inward wrist. The arm was refitted with
wrist roll/pitch/yaw constrained within 0.55/0.45/0.45 rad, and the dart's axis
was reversed in the hand to keep its tip forward. The thumb opening target was
also adjusted to avoid palm collision. Learned force parameters were retained;
the final checkpoint and video use this pose, with validation repeated on it.
The training report preserves the configuration used during training.

- `inspire_grip/validation.json`: long hold, half timestep, independent randomized
  hold/release episodes, unpowered and initially-open negative controls.
- `inspire_grip/trace.json`: time series from the long hold and release.
- `inspire_grip/barrel_contact_audit.json`: verifies which barrel segments contact
  each fingertip.
- `inspire_grip/left_hand_hold_release.mp4`: two camera views of the same simulated
  state, followed by opening and gravity-driven release.
- `inspire_grip/left_hand_grip.png`: close-up still from that rollout.

Simulation results apply to the tested parameter ranges and initial placement.
Transferring to a physical DFQ hand requires measuring the real dart, contact
friction, force sensing, actuator response, and the exact installed hand revision.
The personally recorded throwing motion dataset is still needed for the later
video-to-robot throwing milestone.
