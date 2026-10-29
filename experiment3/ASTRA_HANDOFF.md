# Astra continuation — Experiment 3

Current user choice: **HaMeR**, with WiLoR preserved as a baseline. Both setups and 30-frame pilots are complete. See `HAMER.md` and `configs/reconstruction_backend.json` first. Continue with reconstruction/retargeting judgment; no long training has started.

## Latest HaMeR state (supersedes WiLoR as the selected track)

- Official source pinned to `3a01849f4148352e9260b69bf28b65d1671a4905`, official checkpoint from authors' public Space; all hashes/revisions in `configs/hamer_assets.json`. Separate `.venv-hamer` works. Versions in `configs/hamer_environment.json`.
- `reconstruct_hamer.py` reuses saved detector boxes with the user-confirmed left label. It uses official preprocessing/model, no Detectron2/ViTPose install. Five detector right-label errors are explicitly corrected and recorded. Process-local Arrow exclusion avoids a pandas/Lightning import hang; model source is unmodified.
- **28/30 full meshes**, including all 11 acquisition-window frames. Two initial no-box frames remain absent. Mesh data validated; video decoded. See `research/hamer_validation.json`. This coverage increase includes supervised side correction, so it is not proof that HaMeR beats WiLoR.
- `data/hamer_pilot/` contains raw mesh/joints/MANO/camera predictions and report. `data/hamer_mesh/` contains OBJ meshes and surface overlays. `research/hamer_mesh_3d.html` is a self-contained rotatable 3D viewer with frame slider. `research/hamer_mesh.mp4` is the sparse surface overlay video.
- Run: `.venv-hamer/Scripts/python.exe reconstruct_hamer.py` from experiment3 (or use experiment3-prefixed paths from hackathon), then `.venv-hamer/Scripts/python.exe render_wilor_mesh.py --backend hamer`. The renderer supports both backends despite its historical filename.
- User requested full 3D surfaces because earlier joint-only WiLoR preview was misleading. WiLoR already had 778 predicted mesh vertices; it now also has `research/wilor_mesh_3d.html`, `wilor_mesh.mp4` and OBJ exports. Original inference/results are preserved.
- Do not assume occlusion solved. HaMeR fills hidden geometry using learned priors; curled fingers/contacts remain uncertain and held-presentation geometry is visibly imperfect. Robot pickup still needs object/contact/collision constraints. No dense 421-frame pass, calibrated world reconstruction, robot retargeting or pickup training has occurred.
- A local static preview server is intentionally running at `http://127.0.0.1:8773/hamer_mesh_3d.html` (research directory only, loopback bind). No inference job remains running from this HaMeR setup. The HTML is self-contained and can also be opened locally after stopping the server.

## Verified state

- Workspace/repo: `C:\robotics\KungFu UniTree\hackathon`.
- User video: `C:\Users\Amarnath\Downloads\IMG_4238.mov`. User explicitly confirmed **LEFT hand, UNMIRRORED**. Do not mirror it. Initial plan's right-hand guess was corrected.
- Experiment 3 has a pinned scene with front table at `[0.62,0.10,0.77]` m, surface z=0.795 m; dart x/y `[0.52,0.28]`. Empty hand and physically settled free dart reset work. `check_setup.py` passed at 0.5/0.25 ms.
- Reachable pregrasp is a diagnostic endpoint with a diagonal offset `[0,0.08,0.12]` m from the nominal saved-grasp wrist target. It is **not** a natural fitted trajectory. User specifically dislikes the arm posture and wants a natural video-guided reach.
- Saved closed grip directly on the table fails: best attempted IK ~9.55 mm position error, ~10.34 degrees orientation error, ~5.1 mm finger penetration. Fit a different acquisition pose, lift physically, then transition to the saved dart grip. Do not teleport or suppress collisions.
- WiLoR-mini pinned to `ebec42f94c389070cdd7dda6fd1bf0b4a659c960`; unmodified source under ignored `vendor/WiLoR-mini`. Local `.venv-wilor` works on Python 3.10.11, Torch 2.6.0+cu124. Additional dependencies local; existing Torch read via system site packages. Environment versions in `configs/wilor_environment.json`.
- Checkpoint, detector, MANO and mean parameters downloaded and hashed. Sources/revisions in `configs/wilor_assets.json`. Model files remain local under ignored `models/`. Large transfer required ranged download fallback, implemented in `download_wilor.py`.
- One-frame smoke and **30-frame float32 CUDA pilot completed**. 23 frames have a selected left detection, 18 have all projected landmarks inside the image. Missing at ~0.492, 0.745, 3.745, 4.000, 4.255, 4.492, 4.745 s. Acquisition gap is a material limitation. Median inference 0.076 s/frame, peak CUDA allocated 2.62 GiB. Startup may take minutes/paging; don't kill a load solely for being slow.
- `check_wilor.py` passed schema/timing/finite-data/handedness checks. Overlay video decodes with monotonic PTS. Projected landmarks visually align during approach/presentation; occluded contacts and curled fingers remain uncertain.

## Files to inspect first

1. `PHASE2.md`, `configs/video_annotations.json`, `research/wilor_overlay_sheet.jpg`, `research/wilor_pilot.mp4`.
2. `data/wilor_pilot/report.json` — all detections, selection, flags and timing.
3. `data/wilor_pilot/hand_raw.npz` — `(30,21,3)` wrist-relative joints, source indices and timestamps; missing detections are NaNs. Per-frame NPZs retain upstream camera/mesh/MANO outputs.
4. `PLAN.md`, `README.md`, `research/setup_audit.json`, `env.py` — physical scope and constraints.

## Commands

From hackathon, no reinstall/download needed:

```powershell
python experiment3/check_wilor.py
experiment3/.venv-wilor/Scripts/python.exe experiment3/reconstruct_hand.py
python experiment3/check_setup.py --render
python experiment3/env.py
```

The current inference command reruns only the 30 selected frames. Dense video inference is not implemented. `prepare_video.py` has already prepared the pilot with original PTS, metadata, source hash and color comparison. Baseline normal RGB decode was visually preferred over the orange-cast HLG-to-SDR variant.

## Recommended next work

Review/recover the acquisition gap, potentially with explicit reviewed hand bounding boxes (`predict_with_bboxes` upstream) rather than silently interpolating a learned grasp. A detector miss is distinct from a confident 3D reconstruction. Preserve uncertainty even if manual boxes recover outputs. Reconstruct a smooth wrist/finger reference only after convention/scale/camera checks.

Specific diagnosis from retained detections: the five acquisition frames at 3.745–4.745 s have a detected hand incorrectly classified **right**, while 0.492/0.745 s have no detections. User-confirmed side is left. Reuse the acquisition bounding boxes with explicit left labels in `predict_with_bboxes`, save that as a separately identified correction pass, and review curled-finger/occluded-contact quality. No need to invent boxes or interpolate for that interval as a first attempt.

Video shows hand and some forearm, not shoulder/elbow. It cannot directly supply the natural whole-arm configuration. Retarget using hand/forearm guidance plus explicit elbow/posture preference, joint-limit margins, smoothness and collision constraints. Camera translation assumes a focal length; it is not a measured metric world path. Keep the supported pelvis scope. Fit reach/acquire/lift, execute in physics, and save outcomes before training. Pen hold and final dart grip are different targets.

Preserve active Experiment 2 work and existing dirty files (`PROGRESS.md`, root `README.md`, `inspire_grip.py`, experiment1/2 and dataset outputs). This work only added/edited Experiment 3. No commits, hardware commands or training-policy changes were made. No WiLoR jobs are intended to remain running after this handoff.
