# Phase 2 — timestamped video and WiLoR pilot

Source: the user's local `IMG_4238.mov`. The source was not modified or uploaded. The user confirmed **left hand, unmirrored**. This supersedes the original plan's tentative right-hand interpretation.

Status: setup, one-frame smoke test and 30-frame CUDA float32 pilot completed. Left-hand predictions exist for **23/30 frames**; 18 frames have all 21 projected landmarks inside the image. Missing predictions occur at 0.492, 0.745, 3.745, 4.000, 4.255, 4.492 and 4.745 seconds. The continuous acquisition gap must be handled explicitly before retargeting. Median per-image inference was about 0.076 seconds after model loading, with peak allocated CUDA memory about 2.62 GiB. Model startup is much slower and can page on this machine. These timings are one pilot measurement.

Export validation passed: timestamp order, array shapes, finite selected predictions, NaNs for missing detections, wrist-relative origins, left-hand selection and overlay file existence. Visual review shows usable projected landmarks during approach and held presentation, with uncertain curled-finger estimates and missing acquisition tracking. No 3D ground truth is available. See `research/wilor_validation.json` and `research/wilor_overlay_sheet.jpg`.

The gap has a specific cause: at 3.745–4.745 s the detector finds a hand but labels it **right**, conflicting with the user's confirmed left hand. Those predictions are retained in per-frame exports but excluded from the left track. At 0.492/0.745 s it detects no hand. The next recovery attempt should reuse the acquisition boxes with the confirmed left label via `predict_with_bboxes`, then visually review the new reconstruction; this is more appropriate than interpolating over the gap.

## Prepared reference

`prepare_video.py` extracts 30 original frames at approximately 0.25-second intervals, including the last frame. `data/source_manifest.json` preserves all 421 original presentation timestamps, the source SHA-256, encoded color tags, selected original frame indices and media metadata. There is no frame interpolation or assumed 60 FPS clock. `configs/video_annotations.json` is the reviewed annotation source; a local copy is in `data/annotations.json`.

The timestamp sheet is `research/phase2_timestamps.jpg`. Approximate phases: initial object 0–1 s, approach/preshape 1–2.25 s, occluded acquisition 2.25–4.75 s, lift/wrist rotation 4.75–5.75 s, held presentation thereafter. The acquisition window is not a set of contact labels. Exact contact and liftoff cannot be established from this view. One initial pen-axis annotation is approximate in image pixels; it supplies no metric scale.

`research/phase2_color_comparison.jpg` compares normal decoded RGB against explicit HLG-to-SDR Hable conversion. The latter introduces a stronger orange cast; baseline inference uses normal decoded RGB, which retains useful visible detail. This is a visual choice, not a calibrated color measurement.

## Implementation and reproducibility

The [official WiLoR repository](https://github.com/rolpotamias/WiLoR) links [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) for demonstrations. This pilot pins mini commit `ebec42f94c389070cdd7dda6fd1bf0b4a659c960`. It is a demo front end, not yet the production retargeting pipeline. Asset sources, revisions and hashes are in `configs/wilor_assets.json` after download. Checkpoints come from the authors' Hugging Face Space; MANO/mean parameters use the mini package's public asset source. WiLoR, MANO and Ultralytics retain their respective terms; generated local model files are ignored rather than redistributed.

The separate `.venv-wilor` reads existing CUDA Torch packages through system site packages; additional packages are installed locally with `--no-deps`, without upgrading the throwing environment. `configs/wilor_environment.json` records exact inspected versions. This uses Torch 2.6/CUDA 12.4, newer than upstream's advertised setup. The runner supplies process-local compatibility for legacy NumPy aliases and loads the hash-checked WiLoR checkpoint with explicit metadata deserialization and CPU memory mapping. No upstream source files are edited.

```powershell
# From hackathon; setup is only needed when environment/models are absent.
powershell -ExecutionPolicy Bypass -File experiment3/setup_wilor.ps1
python experiment3/prepare_video.py
experiment3/.venv-wilor/Scripts/python.exe experiment3/reconstruct_hand.py --start 8 --limit 1
experiment3/.venv-wilor/Scripts/python.exe experiment3/reconstruct_hand.py
```

The runner uses float32, one image per call, two Torch CPU threads, and a free-GPU-memory preflight. It preserves every returned detection. Only left-hand detections enter the candidate track; it does not silently relabel detections to match the user. The frame NPZs retain upstream arrays, including projected landmarks and camera translation. `hand_raw.npz` contains wrist-relative 21-joint predictions and original timestamps, with NaNs for missing left-hand detections. `report.json` records omissions, phase flags, latency and peak GPU memory. `research/wilor_pilot.mp4` is a timestamped sparse overlay preview, not a dense reconstruction of all 421 frames.

## What the pilot can establish

Review projected landmarks against visible fingers, especially through closure and wrist rotation. Detection counts alone do not certify 3D accuracy. Camera translation uses an assumed focal length and must not become a metric robot wrist path. Hidden contacts need physical/object constraints. Raw left-hand orientation outputs also require convention checks before robot use.

This video does not show the shoulder or elbow, so it cannot directly specify a natural whole-arm pose. Phase 3 should use the visible hand/forearm trajectory together with robot elbow posture preferences, joint-limit margins, smoothness and collision constraints. The demonstrated pen hold is a motion reference; the saved dart grip remains a separate terminal requirement. No arm motion, pickup training or throwing-policy change is part of this pilot.
