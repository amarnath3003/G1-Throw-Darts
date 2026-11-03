# Why the 200k-step policy still throws low

29 October 2026. The completed run reached **200,704 total steps**. This revision
is an experimental continuation, not a claim of a solved task.

## Measured failure

Audit: `analysis/20261029T154326030486/audit.json`. All 6,070 completed training
throws in the two long runs were inspected through their recorded outcome and
release states. Neither run recorded a board hit. The second run contained
2,737 environment hits and 26 hand recontacts. Video is retained for every throw.

We also executed four **matched, deterministic-policy, randomized-physics**
episodes per configuration, using seeds 73129–73132. Four episodes are a
diagnostic, not a statistically strong performance estimate.

| Configuration | Median release vx | Median release vz | Ballistic board-plane error | Board/bull hits |
|---|---:|---:|---:|---:|
| 100,352-step checkpoint, legacy | 2.78 m/s | 1.04 m/s | 3.59 m | 0/4 |
| 200,704-step checkpoint, legacy | 3.43 m/s | 0.87 m/s | 2.26 m | 0/4 |
| Same 200,704-step weights, wider arc_v2 controls, before retraining | 4.19 m/s | 0.76 m/s | 1.55 m | 0/4 |
| arc_v2 after 12,288-step adaptation, before output calibration | 4.19 m/s | 0.76 m/s | 1.55 m | 0/4 |

These are medians of per-episode measurements; they are not a single synthetic
throw. Predicted plane errors extrapolate through the floor and omit drag;
actual outcomes were floor/environment impacts before reaching the board.

The 200k legacy release requires a median **3.98 m/s upward velocity** at its
measured forward speed, a median deficit of **3.17 m/s**. From those positions,
the unconstrained minimum ballistic speed is about **5.25 m/s**. This lower
bound ignores arm feasibility, dart attitude, aerodynamics and release delay.

The legacy reward `3 exp(-(board_error / 0.8)^2)` has a median value only
0.00116 in that matched evaluation. On many older throws it is effectively zero.
PPO can improve forward progress while receiving almost no reward distinction
between low trajectories. The learned value estimate can be accurate for that
poor reward without the policy becoming an accurate thrower.

## Research informing the change

1. [TossingBot — Zeng et al.](https://arxiv.org/abs/1903.11239) combines a
   physics-derived throwing controller with learned residual corrections.
   We adopt the idea of a physical release-state teacher; our algorithm is PPO,
   not a reproduction of their visual grasping/throwing method.
2. [Learning Accurate Whole-body Throwing — Ma et al.](https://arxiv.org/html/2506.16986v1)
   tracks end-effector position/velocity and handles a release window, rather
   than assuming an open command instantaneously detaches the object. We log
   actual contact-free release and opening delay; we do not implement their
   high-frequency residual policy or convex tube controller.
3. [Ng, Harada and Russell, 1999](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)
   motivates care with reward shaping. Our bounded improvement heuristic is
   **not** claimed to satisfy their potential-based policy-invariance theorem.

## The arc_v2 continuation

The target remains x=3 m, z=1.73 m and the inner bull remains 6.35 mm radius.
Gravity, dart mass/inertia, finger forces, friction, actuator torque ceilings,
collision checks and real release rules are unchanged. The pelvis remains
supported. A launch impulse or attached dart would invalidate this experiment.

For each candidate flight time T from 0.20 to 1.20 seconds, compute the velocity
that would connect the current dart COM p to the goal g under gravity:

`v_required(T) = (g-p)/T + [0, 0, 4.905 T]`

Find the nearest member of that family to the actual dart velocity v. With
velocity error e in m/s, use `quality = 1/(1+(e/2)^2)`. This gives poor throws a
graded signal and permits different viable arcs, without rewarding arbitrary
height or speed. It is still an approximate teacher: the finite time range,
COM-to-goal approximation and omitted drag/attitude limit its accuracy.

Changes:

- Once at valid physical detach: **24 × quality**, plus up to **2** for
  nose-to-velocity alignment. This replaces the old nearly-zero prediction term.
- Before detach: **6 × change in quality** per control step. Holding a stationary
  configuration cannot repeatedly accumulate this reward. No readiness rewards
  are paid during free flight.
- At a valid flight's terminal board/environment/timeout outcome: up to **8**
  from measured closest tip-to-bull approach, `8/(1+(distance/0.5)^2)`.
  This uses the actual aerodynamic simulation, not predicted success.
- Physical board/bull rewards and invalid-contact penalties remain authoritative.
  Robot hits, unintended grip contacts and recontacts receive no terminal
  approach bonus. Positive shaping rewards still need monitoring for shortcuts.
- Residual bounds expand from `[.45,.25,.30,.65,.20,.50,.20,.15,.12]` to
  `[.70,.35,.45,1.0,.30,.65,.30,.20,.18]` rad for the same nine joints.
  Per-20-ms residual slew expands from .04 to .06 rad. This changes the action
  interpretation but keeps checkpoint tensor shapes and observations compatible.
  It is **not** proof that a successful release is reachable; measured rollouts
  must establish that. Joint target clipping and original torque limits remain.
- Preserve actor weights and observation normalization, reset the critic and
  its optimizer moments, use learning rate 1e-4, entropy coefficient .015 and
  action std .65 to explore beyond the flat-throw local optimum. Actor hashes
  verify that setup did not overwrite learned motion. Critic reset is a deliberate
  warm restart, not exact optimizer continuation.

### Second diagnosis: clipped exploration

The short 12,288-step adaptation did not promptly reduce the velocity gap. At
the 200k policy's reset state, raw Gaussian action means were approximately
`[-.85,1.39,-1.77,2.78,2.00,2.30,2.72,-1.14,-1.44,1.15]`, while actions are
clipped to [-1,1]. Setting std=.65 alone does not restore useful exploration in
an output centered near 2.8: almost all samples still execute +1.
Matched deterministic reevaluation after that pilot produced the same physical
trajectories as the unadapted wider-control baseline: its changed network
outputs still collapsed to the same clipped commands.

The optional **`--desaturate-actor`** warm-start step therefore scales each
output-layer row and bias by `min(1, .8 / max_abs_reset_mean)` across the reset
workers. It preserves the learned actor feature network and action signs, but
deliberately changes saturated output magnitudes. Output-layer Adam moments
are cleared. Exact before/after means, row gains and preservation checks are
written to `adaptation.json`. No action weights are silently reset. This is a
one-time calibration; ordinary subsequent resumes must omit the flag.

A local rigid-hand Jacobian check also found insufficient velocity authority
at one existing release pose with the old reference and ±3 rad/s residual rates.
That is a warning about the inherited motion, not a general impossibility
proof. Better release timing/posture must still be learned; if progress stalls,
the next change should address motion feasibility rather than rewards alone.

The calibrated continuation is `runs/20261029T155558346875`, initialized from
the original 200,704-step checkpoint with 100,000 additional requested steps,
four workers and 256-step rollouts. CUDA policy and optimizer tensors are
verified. At 2,048 new steps it had 81 completed exploration episodes and no
board hits; the median velocity gap was 3.83 m/s, worse than the inherited
deterministic policy. Calibration restores exploration, not instant performance.
The unmodified 200k checkpoint and all earlier videos remain available.

An evaluation seeding bug was also found: loading PPO could restore its saved
seed after the vector environment had been seeded. The loader now explicitly
receives `seed=args.seed`. The original matched comparison in `audit.py` is
unaffected because it explicitly resets each episode with its listed seed.
`runs/20261029T155608819943/evaluation.json` is retained but marked invalid as
a held-out-seed evaluation; it actually used seed 20261029 rather than 73129.

The subsequent checkpoint integration check in `runs/20261029T160123989927`
loaded the calibrated 202,752-step periodic checkpoint with its paired normalizer,
inherited `arc_v2` automatically and completed a CUDA evaluation using the fixed
seed override (73129). It was a failure, not a bullseye. The physical reward and
collision tests pass for both legacy and revised variants. At 7,168 new training
steps the calibrated run's rolling velocity gap had fallen from 3.85 to about
3.33 m/s; that is recovery within the new exploration run, still worse than the
old deterministic checkpoint's 2.63 m/s matched-evaluation gap. These quantities
also differ in stochastic versus deterministic action selection and should not
be treated as a controlled performance comparison.

New telemetry includes required vz, velocity gap to the arc family, predicted
board error, actual peak height, closest tip approach, actual board-plane
crossing, opening-command-to-detach delay and nose alignment. Runtime metrics
report a rolling 100-episode window. Training rewards across legacy and arc_v2
are not comparable; compare physical outcomes and matched evaluation seeds.

The checkpoint viewer and trainer select the saved environment variant. Old
checkpoints default to legacy. A process already running the old viewer must be
reopened to load these code changes.

## Validation and next decisions

Reward tests verify that increasing upward velocity toward a valid arc improves
guidance and an exact ballistic arc maximizes it. Gymnasium contract, deterministic
reset and genuine MuJoCo collision-gate fixtures are checked independently.
Synthetic collision fixtures are not learned bullseyes.

Run a short CUDA adaptation and evaluate again before allocating another long
run. First criterion: shrink the velocity gap/vertical deficit while preserving
clean detach. Next: reach the physical board. Only then tighten accuracy through
the existing center rewards and larger held-out evaluation. If the velocity gap
plateaus, inspect joint saturation and actual arm reachability before simply
adding more steps. A later control redesign can add a feasible release-state
reference, richer phase control or a learned demonstration prior. Do not move
the target or change dart physics to disguise poor throws.

Every completed training and evaluation swing continues to save its video and
states. No autonomous pickup, unsupported balance or real-hardware readiness
is implied by this revision.
