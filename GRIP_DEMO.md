# Contact grip and gravity release

Run from `hackathon/`:

```powershell
python grip_demo.py
```

This opens a MuJoCo viewer and a joint-control panel. The robot starts holding a
vertical, tip-down dart. Its right wrist points toward the bullseye. Use the
seven arm sliders to edit the joint targets, or change the inward gripping force.
The pose is authored directly; no Gary footage or video reconstruction is used.

| Control | Action |
| --- | --- |
| **O / Open** | Open to 72 mm so the barrel and flights clear; dart falls |
| **C / Close** | Apply inward force to both fingers |
| **R / Reload** | Reset the simulation and place the dart at the current hand pose |
| **D / Run demonstration** | Reload, hold for three simulation seconds, open |
| **Space** | Pause/resume physics |
| **1 / 2 / 3 / 4** | Grip close-up / drop / robot / room cameras |
| **Restore bullseye pose** | Restore the tested joint targets and reload |

Keyboard shortcuts apply in the MuJoCo window; panel buttons provide the same
actions. Closing after a drop closes the empty gripper. Reload is an explicit
episode initialization, not pickup or retrieval. Large manual joint changes can
lose the grasp; use Restore to return to the tested setup. The rendered movie
plays at simulation speed; live simulation speed depends on the computer.

## Mechanics

- A separate demo is composed from `scene.xml` and the bundled G1 at runtime.
  It does not replace the room scene or the original rigid-hand model.
- The pelvis is fixed to the world, with a visible gold support fixture. The
  remaining body joints use torque-limited PD servos with bias compensation.
  This is a supported manipulation test, not a standing-balance result.
- The right rigid hand is replaced by a custom two-finger parallel gripper.
  Each finger has a sliding joint and a force-limited motor. Default squeeze is
  **0.8 N per finger**, with a 4 N actuator limit. The control panel exposes 0–2 N.
  The custom mount is 120 g and the two fingers/pads total 56 g. The original
  wrist-link inertia is retained; this assembly is not a hardware-identified hand.
- The pads use approximate friction (sliding coefficient 0.7) and MuJoCo compliant
  contact. This is a functional contact prototype, not an Inspire or Dex5 model.
- The existing **26.7 g** dart retains its profile, center of mass, inertia,
  collision shapes and explicit aerodynamic forces. It has a free joint at all
  times. There is no dart weld, grasp attachment, scripted fall, upward support
  force or per-step position overwrite. Only reset places the dart in the hand.
- Finger normal pressure supplies friction against gravity. Opening moves both
  fingers clear of the flights; gravity, contact and the existing air-load model
  determine the drop and landing on the teal mat.
- Reducing squeeze alone is not the same as opening. The barrel slips, but the
  wider flight can catch on the still-closed jaws. This appears in the negative
  controls and is a reason to command enough clearance during release.

## Reproduce the evidence

```powershell
python grip_demo.py --check
python grip_demo.py --render
```

The viewer needs Tkinter (included with the standard Windows Python installer).
Rendering additionally needs FFmpeg on PATH. Core simulation uses the existing
`requirements.txt`; no new Python package is needed.

Outputs in `grip_demo/`:

- `grip_release.mp4`: two-view hold/open/drop recording.
- `holding.png`, `falling.png`, `landed.png`: recorded simulation frames.
- `validation.json`, `trace.csv`: measurements and assertions.
- `pose.json`: joint targets in radians/degrees and manual-pose provenance.

The checked default pose held for 10 seconds with 1.125 mm maximum vertical
change, approximately 0.8 N normal force per pad and 0.261927 N upward contact
force, matching the dart's weight. After opening, it dropped 0.4414 m in 0.3 s;
fitted free-flight acceleration was -9.8081 m/s². It landed on the mat.
Aim error at the end of the hold was 0.163 degrees.

Additional checks halve the timestep to 0.25 ms, vary squeeze to 0.4/1.2 N per
finger, and confirm barrel slippage with zero squeeze or frictionless pads.
An initially open gripper gives free fall. These are simulation checks, not
hardware calibration or evidence of sim-to-real accuracy.

Next integration: replace the custom gripper with the chosen articulated hand,
calibrate its contact/actuator parameters, and keep the same hold/release checks
before adding moving-arm releases or residual PPO.
