# Progress check — 29 October 2026

The latest saved policy is moving in the right direction, but it still cannot
reach the board. This conclusion uses measured free-flight paths as well as
reward/ballistic diagnostics.

## Matched checkpoint comparison

Eight identical randomized-physics seeds (74101–74108), deterministic policy
actions, original supported G1/Inspire hand and dart physics. Every evaluation
throw is retained as `eval_*/00000*/swing.mp4` with states and results.

| Median metric | Original 200,704-step policy | Revised 251,904-step policy |
|---|---:|---:|
| Forward release velocity | 3.56 m/s | 3.44 m/s |
| Upward release velocity | 0.70 m/s | 1.38 m/s |
| Actual closest tip-to-bull distance | 1.37 m | 1.21 m |
| Peak dart COM height | 1.45 m | 1.52 m |
| Velocity gap to a ballistic target arc | 2.71 m/s | 2.18 m/s |
| Board hits / bullseyes | 0 / 0 | 0 / 0 |

Actual closest approach improved about 12%; this is a 3-D minimum distance during
flight, **not** a board-impact error. All these throws hit the environment before
the board. Eight trials are a small diagnostic, not a strong success-rate estimate.
The intermediate 227,328-step policy was worse in actual approach than the
original, so progress has not been monotonic. All four checkpoints had 8/8
clean releases in this deterministic comparison.

The latest policy still needs approximately **3.98 m/s upward velocity** at
its measured forward speeds; its median actual value is only 1.38 m/s. Typical
peak height is also below the bull's 1.73 m height. This remains a substantial
motion/release feasibility problem. Higher shaped return alone is insufficient.

## Stochastic training evidence

Across 1,646 archived completed training throws, the first 200 versus the last
200 improved from 65% to 98.5% clean flights. Median upward release speed among
clean flights increased from 0.84 to 1.20 m/s, and closest approach from 1.88 to
1.42 m. All 1,646 missed the board. Training windows include exploration noise;
they are separate from the matched deterministic evaluation above.

## Run recovery

The trainer process was absent, with no final policy/summary. Its latest valid
paired checkpoint was 251,904 steps. The cause of process exit is unknown.
The checkpoint ZIP passed integrity checking, and training was resumed without
repeating critic reset or output calibration.

Active continuation: `../../runs/20261029T163135827530`.
Remaining requested steps: **49,152**, to reach the original planned rollout-rounded
total of **301,056**. Policy and optimizer tensors were verified on CUDA, with
four CPU MuJoCo workers. Startup/output/error logs and recovery metadata are in
`../../runs/progress_resume.json`. Each completed attempt continues to save video.

Raw snapshot and all results: [progress.json](progress.json).
Plot: [progress.png](progress.png).

Recommendation: let this bounded continuation finish and repeat the matched
evaluation. If upward-speed deficit and measured approach plateau, investigate
release posture, arm acceleration authority and release timing before adding
another long run. No additional physics/reward changes were made during this check.
