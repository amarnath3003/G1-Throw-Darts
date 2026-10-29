# Experiment 3: tabletop pickup — step 1

**Selected reconstruction backend: HaMeR.** Setup and the 30-frame full-mesh pilot are complete; see `HAMER.md`. WiLoR is preserved alongside, including its full-mesh viewer. Current continuation instructions are in `ASTRA_HANDOFF.md`.

Phase 2 setup and the 30-frame WiLoR pilot are also complete. See `PHASE2.md` for results and `ASTRA_HANDOFF.md` for continuation instructions. The pilot has a missing acquisition interval that needs review before retargeting.

Implemented: an experiment-local supported G1 scene, empty-hand/tabletop-dart reset, and a reproducible kinematic/clearance audit. This is not a pickup controller or trained policy.

The table is in front of the robot (world +X), centered at `[0.62, 0.10, 0.77]` m, with a 0.50 × 0.40 m surface at z = 0.795 m. The dart starts at x/y = `[0.52, 0.28]` m, near the robot-facing side and slightly toward the left hand. This arrangement supports a forward reach resembling the egocentric demonstration; video-to-robot calibration has not yet been performed. The original right-side equipment table, tray and prop darts are not part of this experiment's pickup surface.

## Run

From the hackathon directory, using the existing Python with MuJoCo and SciPy:

```powershell
python experiment3/env.py
python experiment3/check_setup.py --render
```

The first command opens the empty-hand scene. The second checks resets and offline IK at 0.5 ms and 0.25 ms timesteps and writes `research/setup_audit.json`, `reset.png`, `pregrasp.png`, and `terminal_grip_on_table.png`. IK pictures are diagnostic poses, not physically executed trajectories. A successful exit verifies the stationary reset and reachable, table-clear pregrasp; it does **not** mean the closed grasp passes the tabletop clearance audit.

On a fresh checkout, first run:

```powershell
python experiment3/setup_scene.py
```

The builder refuses to overwrite an existing model. Explicitly archive the existing snapshot before intentionally rebuilding. It reads the shared scene/assets once, recompiles static collision geometry after relocation, and stores a self-contained binary model plus a copied grasp configuration under `configs/`. The runtime loads these pinned files, verifies their hashes, and does not import the active throwing environment. Source and asset hashes are recorded in `configs/scene.json`. Binary snapshots and the generated XML are local/ignored; rebuilding from changed shared sources creates a different snapshot. MuJoCo binary compatibility is version-specific; the recorded version is 3.10.0.

## Reset and scope

- Reset opens the left hand, initializes the robot in its existing raised arm pose, and puts the free dart 1 mm above the table using its lowest collision geometry point.
- Two seconds of torque-controlled physics settle the dart. There is no weld, grasp attachment, in-hand placement, or object state edit after settling. Contact margins explain the small positive geometric gap reported at rest.
- The unused right hand retains its default open state. The robot's supported pelvis and original joint/actuator limits are retained. The relocated table and decorative-object removal affect only the snapshot.
- Reset uses the existing torque ceilings with position/velocity feedback and gravity compensation. It is a stationary setup controller, not the force-regulated pickup controller planned for step 3. Air loads are omitted for this stationary audit; pin and restore the dart aerodynamic model before transport/throw integration.
- IK respects independent arm joint limits and hand mimic coupling. It uses the settled dart pose and the existing hand-relative grip transform. The pregrasp target is 120 mm above and 80 mm toward the left of that transform, allowing a diagonal acquisition approach. It audits hand/table signed distances, fingertip vertical extrema and other penetrating contacts. Existing stationary foot/floor contact pairs are reported but excluded from the new-collision gate. No collision-free path or dynamic reach is claimed from endpoint IK.

## Clearance finding and next step

The final front-table setup passed reset and pregrasp checks at both timesteps. The diagonal pregrasp IK residual is below 1 micrometre and 0.001 degrees, with approximately 124 mm minimum hand/table separation and no new penetrating contacts. Repeated settled resets match exactly. These are endpoint/initialization results, not dynamic trajectory accuracy.

The attempted closed grasp at the tabletop fails: approximately 9.55 mm position error, 10.34 degrees orientation error, and 5.1 mm middle-finger/table penetration. Those clearance values describe the best attempted IK pose, not a successfully fitted grasp. This failure is retained in the audit and does not invalidate the reachable empty-hand starting setup.

The saved closed grip must be audited separately from approach reachability. In the earlier side-table test it intersected the table by approximately 10–14 mm at the index/middle fingers. See the current front-table measurements in `research/setup_audit.json`; a negative signed distance means intersection. Do not replay an intersecting pose as a valid pickup or weaken collision geometry to pass it.

Step 3 must fit a feasible acquisition configuration and physically lift the dart before transitioning into the saved throwing grip. A raised cradle would be a separately labeled assisted curriculum, not flat-table pickup. WiLoR annotations remain step 2; no reconstruction or training has started here.
