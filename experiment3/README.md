# Experiment 3 · Tabletop pickup

The pickup pipeline reconstructs the hand motion, fits a seven-joint arm
reference, and rehearses approach and lift in MuJoCo. The contact controller
uses the original hand geometry and torque limits. The residual PPO environment
starts with an empty hand and a free, settled dart. Its success gate checks
table clearance, loss of table contact and a sustained grasp.

The [HaMeR-guided arm rehearsal](../progress_videos/10_g1_hamer_guided_arm_rehearsal.mp4)
executed at 0.5 ms and 0.25 ms. The later pickup work recorded an individual
nominal policy rollout with about 99 mm of lift. The
[timestep comparison](../progress_videos/13_pickup_timestep_comparison.mp4)
shows the sensitivity of the fitted contact controller.

## Inspect the scene

From the repository root:

```sh
python experiment3/setup_scene.py
python experiment3/check_setup.py
python experiment3/env.py
```

The setup script builds the experiment-local MuJoCo snapshot. The setup audit
checks repeatable reset and clear pregrasp poses. See [HaMeR](HAMER.md) and
[WiLoR](PHASE2.md) for reconstruction details.
