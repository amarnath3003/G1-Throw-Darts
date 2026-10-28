# Progress — 28 October 2026

## Completed

- Built and dressed the MuJoCo room, G1, dartboard, throwing line, equipment
  station, lighting, and collision geometry. Bull height is 1.73 m and the board
  face is 2.37 m from the throwing line.
- Added a 26.7 g dart based on the James–Potts experimental reference, with
  measured center of mass, inferred inertia, detailed collision geometry,
  explicit air loads and numerical validation. Contact and aerodynamic
  parameters still require project hardware calibration.
- Downloaded 119 dataset videos and three coaching sources for local inspection.
  Prepared 20 body-motion candidates and four qualitative references, with a
  local review gallery, provenance records, checksums and approval gating.
- Approved all six Group A throws, D01–D06, and Gary side-view references R03–R04.
  The six body clips have 30 FPS GVHMR copies and source-frame timing maps. Gary
  clips remain qualitative because their timing and full-body visibility are
  unsuitable for physical velocity supervision. Downloaded third-party media is
  excluded from Git; selection metadata and processing tools are included.
- Built a separate supported grip demo with a manually authored arm pose toward
  the bullseye, a custom two-finger gripper, adjustable inward force, seven arm
  joint sliders, open/close controls, reset and a demonstration cycle.
- Kept the dart free throughout holding and release. Contact friction supplies
  the hold; opening the fingers lets gravity produce the drop and mat landing.
  Direct joint editing was chosen for this milestone; no Gary motion was used
  to generate the pose.
- Recorded three application-window videos showing the room, controls, and
  physical hold/release. All three H.264 files fully decode and were visually
  inspected. The recorded controls are driven by scripted backend target edits.

## Validation evidence

| Check | Result |
| --- | --- |
| Dart numerical checks | Passed; see `dart_validation.json` |
| Selected video bundle | Eight approved clips; see `video_dataset/catalog/selected_validation.json` |
| Supported hold | 10 s, maximum height change 1.125 mm |
| Pad normal force | Approximately 0.8 N per pad |
| Upward contact force | 0.261927 N, matching the dart's weight |
| Wrist aim error after hold | 0.163 degrees |
| Drop after opening | 0.4414 m in 0.3 s |
| Fitted free-flight acceleration | -9.8081 m/s² |
| Contact and timestep checks | Passed at 0.4/1.2 N squeeze and half timestep |
| Negative controls | Barrel slips without squeeze or pad friction; initially open jaws give free fall |

Detailed traces and checks are in `grip_demo/validation.json` and
`grip_demo/trace.csv`. Zero squeeze can let the wider flight catch on closed
jaws after the barrel slips, so release opens far enough to clear the flights.

## Recordings

1. [Room and environment — 23 s](progress_videos/01_room_and_environment.mp4)
2. [Joint and grip controls — 25 s](progress_videos/02_joint_and_grip_controls.mp4)
3. [Physical hold and gravity release — 17 s](progress_videos/03_physical_hold_and_release.mp4)

These show the current supported manipulation prototype. A successful throw at
the board has not been implemented. The custom sliding fingers are an initial
grip mechanism; the chosen five-finger hardware model and unsupported standing
are later milestones. Sim-to-real accuracy is not established.

## Run

```powershell
python -m pip install -r requirements.txt
python view_scene.py
python grip_demo.py
python grip_demo.py --check
python grip_demo.py --render
python record_progress.py --clip all
```

Recording needs Windows, Tkinter and FFmpeg/FFprobe on PATH. Detailed controls
are in [GRIP_DEMO.md](GRIP_DEMO.md); dart assumptions and calibration needs are
in [DART_PHYSICS.md](DART_PHYSICS.md).

## Next milestones

1. Collect personally recorded footage, as required by the challenge.
2. Reconstruct and review approved body motion with GVHMR/GMR, resolving physical
   timing before using velocity targets.
3. Install and calibrate the chosen articulated hand, retaining contact-based
   hold/release validation.
4. Add moving-arm releases, board impact/scoring, residual PPO and success-rate
   evaluation, then tackle unsupported standing.
