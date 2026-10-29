# Progress check: higher arc, worse overall reach

29 October 2026. Training remains active; no environment or policy parameters
were changed during this audit. All evaluation attempts have videos and states.
Full numerical results and checkpoint hashes: [progress.json](progress.json).

The evaluated recent checkpoint is **229,376 total steps** (28,672 adaptation
steps). It was selected before the evaluation; the trainer continued independently.

Four deterministic-policy episodes per checkpoint used exactly the same
randomized-physics seeds, 73129–73132. This is a small diagnostic sample.

| Physical metric (median) | Original 200,704 | Early calibrated 202,752 | Recent calibrated 229,376 |
|---|---:|---:|---:|
| Forward release speed | 3.43 m/s | 3.88 m/s | 2.21 m/s |
| Upward release speed | 0.87 m/s | 0.64 m/s | 1.67 m/s |
| Peak dart COM height | 1.46 m | 1.45 m | 1.55 m |
| Actual closest tip-to-bull distance (lower is better) | 1.37 m | 1.32 m | 1.69 m |
| Ballistic velocity-family gap (lower is better) | 2.63 m/s | 2.62 m/s | 2.69 m/s |
| Valid board hits / bullseyes | 0 / 0 | 0 / 0 | 0 / 0 |

**Assessment:** upward motion improved, but lost forward speed more than offsets
that gain. Overall aiming/reach is worse in these matched tests. This does not
establish statistically reliable regression or future convergence; it does rule
out claiming an observed accuracy improvement from this comparison.

Within stochastic training, the first-to-latest 100-episode windows show fewer
invalid releases, smaller velocity gaps and closer flight paths. That is recovery
from the deliberately disturbed warm start, not proof of improvement over the
old policy. Training and deterministic evaluation must not be conflated.
See [training_trend.png](training_trend.png).

A slightly later archive read found 1,043 completed training episodes, including
860 uninterrupted forward flights: zero board hits, and none of those flights
had a peak COM height at or above 1.73 m. Maximum COM height was 1.665 m. The
bullseye remains at 1.73 m, with no target/physics changes during this check.

Next diagnostic priority: establish a physically achievable combination of
forward and upward release velocity, accounting for arm rate/torque limits and
opening delay. The inherited motion and residual authority may constrain that
combination. Simply rewarding height more strongly risks sacrificing more
forward speed. Keep the original 200k and early calibrated checkpoints as
comparators rather than treating the newest checkpoint as automatically best.
