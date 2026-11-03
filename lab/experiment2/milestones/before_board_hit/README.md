# Before board contact — 30 October 2026

The arc_v7 PPO continuation finished at 409,708 total steps. Policy inference
and optimization used CUDA; four CPU MuJoCo workers collected rollouts.

The best fixed-seed physical evaluation, at 409,356 steps, produced 12/12 clean
flights, zero valid board hits and zero bullseyes. Mean penalized closest
tip-to-bull distance was 0.595409 m. This is closest approach during flight,
not impact error on the board.

This commit preserves the environment, trainer and evaluation sources archived
by run `20261030T125811941261`, before subsequent joint-reference calibration.
The original left five-finger hand, contact-held free dart, physical parameters
and supported pelvis remain part of the setup. Every completed throw was recorded.

Generated checkpoints and rollout collections remain local under
`experiment2/runs/`; they are excluded from Git. The run's source manifest and
best evaluation summary are copied here for provenance. Existing revision
checks cover action sensitivity, reference derivatives, physical invariance,
release timing and physical checkpoint selection. No board-hit skill is claimed
at this milestone.
