# HaMeR reconstruction track

The user selected **HaMeR** after clarifying the distinction from HaWoR. HaMeR is now the intended reconstruction backend; WiLoR's source, environment, models, pilot predictions and previews remain available for comparison. HaWoR was inspected/cloned during model identification but is not the selected pipeline.

Status: **installed, checkpoint verified, CUDA float32 inference and surface rendering complete**. 28/30 pilot frames have full meshes (778 vertices/1,538 triangles each); only the two initial no-box frames at 0.492/0.745 s are missing. All 11 acquisition-window samples have predictions after correcting the five wrong-side detector labels using the user's confirmation. Peak allocated CUDA memory was 2.59 GiB. Mesh shape/finite-data/timestamp/handedness/export checks passed and the surface video decoded successfully. `research/hamer_validation.json` records the results. This is a 30-frame pilot, not dense inference across all 421 frames.

Visual review: the open-hand approach surface follows the visible silhouette reasonably; acquisition now has a surface estimate, while held-presentation curled-finger geometry remains uncertain and sometimes visibly mismatched. A filled mesh has not solved the occlusion/contact problem. Review those frames before using them as robot targets.

Both [HaMeR](https://geopavlakos.github.io/hamer/) and WiLoR output a complete MANO hand surface. The previous WiLoR video displayed only projected joints; the saved NPZs already contained 778 mesh vertices per prediction. `render_wilor_mesh.py` now exports surfaces and interactive 3D views for either backend.

## Run

From `hackathon`, after `setup_hamer.ps1` has prepared the environment and verified model assets:

```powershell
experiment3/.venv-hamer/Scripts/python.exe experiment3/reconstruct_hamer.py
experiment3/.venv-hamer/Scripts/python.exe experiment3/render_wilor_mesh.py --backend hamer
```

The HaMeR runner uses the official [implementation](https://github.com/geopavlakos/hamer) pinned to `3a01849f4148352e9260b69bf28b65d1671a4905`, official `ViTDetDataset` crop preprocessing and the official checkpoint from the authors' Hugging Face Space. Checkpoint SHA-256 is verified against upstream metadata. Asset revisions/hashes are recorded in `configs/hamer_assets.json`. The existing MANO model is copied locally; nothing is installed into WiLoR or the throwing environments.

The separate `.venv-hamer` reads existing CUDA Torch via system site packages. Only the inference dependencies are added locally. Detectron2/ViTPose are unnecessary for this pilot because it reuses the saved detector boxes. Renderer objects in HaMeR are disabled; the preview exporter rasterizes the saved mesh and supplies an interactive Plotly surface. This avoids requiring an offscreen OpenGL context on Windows. A process-local `sys.modules['pyarrow']=None` prevents an observed hang in the optional Arrow backend during pandas/Lightning import. No Arrow operations are needed for inference.

## Input fairness and uncertainty

The user confirmed a single left hand, unmirrored. Every reused detector box is explicitly assigned that known left side before HaMeR's crop/forward pass. The original detector label is retained. This fixes the **input handedness** during the five acquisition frames previously labeled right. It does not demonstrate that HaMeR outperforms WiLoR: the original WiLoR pilot used the detector's predicted side, so a fair comparison would rerun WiLoR with the same corrected-side boxes.

Closed fingers hidden behind the hand or object are inferred from a learned prior. A complete mesh can look plausible while its hidden contacts are wrong. Retain acquisition uncertainty, check visible projections and temporal behavior, and fit the robot pickup with object/contact/collision constraints. HaMeR does not reconstruct the pen or recover calibrated world coordinates/camera motion. It cannot directly provide an observed elbow/shoulder trajectory from this clip.

## Outputs

- `data/hamer_pilot/`: per-frame predicted vertices, joints, camera values, canonical MANO rotations/shape, wrist-relative joint trajectory, timestamps and report. Canonical MANO parameters are for the right-hand template; camera-oriented vertices/joints are mirrored to left as in the official demo. Do not treat the canonical rotation matrices as validated robot wrist rotations.
- `data/hamer_mesh/`: OBJ meshes and projected surface frames.
- `research/hamer_mesh_3d.html`: self-contained rotatable 3D surface with a frame slider; wrist-centered coordinates and estimated shape, not calibrated world motion.
- `research/hamer_mesh.mp4`: sparse full-surface overlay preview preserving selected-frame timing.
- `research/hamer_mesh_preview.png`: representative mesh overlay.
- WiLoR equivalents use `wilor_` names and remain available. `render_wilor_mesh.py --backend wilor` only renders existing predictions and does not rerun WiLoR.
