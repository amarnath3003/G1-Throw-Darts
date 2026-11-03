# Experiment 4 · Video-guided throwing

A left-handed two-swing reference is reconstructed with GVHMR and fit to the
G1 left arm. The fit preserves the two pullbacks and adds shoulder-relative
clearance constraints for the elbow, wrist and dart. The video-derived
reference also drives the residual PPO environment.

## Measured results

The fitted trajectory passed both pullback checks and clean-release checks at
0.5 ms and 0.25 ms. Minimum wrist and elbow clearance outboard of the
shoulder was about 4.1 cm and 1.6 cm respectively. Both replays missed the board.
Initial PPO validation logged 57 clean releases, seven slips and zero board
hits in 64 simulated throws.

## Pipeline

1. `prepare_reference.py` trims the source and preserves decoded timestamps.
2. GVHMR estimates the SMPL-X sequence. `reconstruct_arm.py` exports the arm trajectory.
3. `retarget_arm.py` fits the G1 reference, and `check_throw.py` verifies the physics.
4. `video_env.py` and `train_video.py` connect the reference to the shared PPO trainer.

The exact reference recording and generated reconstruction inputs are kept in
the local project archive. The video preprocessing tools require FFmpeg,
FFprobe and an independently installed GVHMR environment.
