# Experiment 2 — PPO learns the throw

Experiment 1 is stopped and its recordings are retained. This experiment uses
its restrained shoulder–elbow–wrist motion as an initialization, not as a
successful demonstration. PPO must learn corrections and when to open the hand.
The objective is a clean, tip-first impact within the **6.35 mm inner bull**.
No trained bullseye performance is claimed by the initial implementation.

The original CUDA continuation finished at **200,704 steps**, with no board hits.
The failure audit and researched reward/control revision are documented in
[REWARD_REVISION.md](REWARD_REVISION.md). `arc_v2` is a separate, checkpoint-aware
environment variant; old runs retain the `legacy` interpretation.

To adapt the old actor to the new trajectory reward (do not reset the critic
again on ordinary subsequent resumes):

```powershell
python experiment2/train.py --resume experiment2/runs/OLD/policy.zip --variant arc_v2 --reset-critic --desaturate-actor --learning-rate 0.0001 --entropy 0.015 --exploration-std 0.65 --steps 12288 --workers 4 --rollout 256 --randomize
python experiment2/train.py --evaluate experiment2/runs/NEW/policy.zip --episodes 20 --seed 73129 --randomize
python experiment2/train.py --resume experiment2/runs/NEW/policy.zip --steps 100000 --workers 4 --rollout 256 --randomize
```

Variant selection is inherited automatically from the source checkpoint's
`config.json` for both resume and evaluation. Keep that file with the model and
normalizer. `progress_metrics.json` reports physical trajectory metrics over the
last 100 completed episodes; a higher reward alone is insufficient evidence of
an improved throwing skill. `adaptation.json` records critic/exploration changes
and verifies preservation of the actor feature network during setup. Output
desaturation deliberately changes the final action layer and is logged explicitly;
omit it on subsequent resumes. `audit.py --evaluate`
reproduces the historical comparison and saves all evaluation videos.

## What PPO controls

At 50 Hz, a 128×128 MLP receives 73 state observations: arm/waist positions and
velocities, finger state and contact forces, dart position/velocity/orientation,
target, phase, recent actions, and release state. Nine actions modify seven left
arm joints and waist yaw/pitch. The tenth requests release; once requested it
cannot be undone during that throw. The earliest command is at 20 ms. This is
state-based residual PPO, not a video-conditioned policy or residual torque PPO.

Residuals have per-joint amplitude and slew bounds. Torque tracking preserves
the original actuator ceilings. The original left Inspire DFQ geometry, finger
couplings, 26.7 g free dart and existing aerodynamic forces are used. The grasp
is placed only at reset. There are no attachment constraints, launch impulses,
or in-flight edits. A 0.25 s settling period must establish three contacts.

The pelvis is still supported. Small hip counter-movements come from the prior;
whole-body balance is not learned in this experiment. The prior is implemented
in `experiment1_throw.make_simple_plan`; Experiment 1's optimization/sweeps are
not run by the trainer. Personally recorded grip photographs still inform the
grasp; personally recorded throwing-motion learning remains a separate milestone.

## Rewards and protections against shortcuts

| Event | Reward / rule |
|---|---|
| Actual board impact | +20, plus up to +20 for proximity to center |
| Inner bull | Additional +100; requires valid release, forward orientation and contact within 3 mm of the tip site |
| Clean release | +2 once; all hand links must be clear for 10 ms |
| Predicted impact at release | Up to +3 once from a rough ballistic estimate; never counts as a hit |
| Forward free flight | At most +2 per episode, based on new forward progress toward the board |
| Slip before command | −15, terminate |
| Robot collision, unintended grip contact, or hand recontact | −15, terminate |
| Backward throw or drop with forward speed below 0.3 m/s | −12, terminate |
| Floor, wall, surround or other environment impact | −8, terminate |
| No release by 1 s | −12, terminate |
| Numerical instability or arm/waist joint-limit violation above 0.03 rad | −25, terminate |
| 2.5 s timeout | −8, truncate |

Small action-size, action-change and finger-overspeed penalties discourage
violent motion. Distance is only a bounded shaping signal after release, so
carrying the dart forward, throwing it sideways or repeatedly opening the hand
cannot accumulate distance rewards. A contact with the board body is identified
by its actual collision geometry, not by assuming it belongs to the world body.
Invalid robot contacts take precedence over simultaneous board contacts.

The ballistic estimate omits aerodynamics; measured simulated contact is the
final authority. Board penetration and sticking are not modeled, so success
means a valid tip impact, not a dart remaining embedded in sisal.

## Run

From `hackathon`, using the project's Python environment:

```powershell
python -m pip install -r experiment2/requirements.txt
python experiment2/check.py
python experiment2/train.py --steps 256 --workers 1 --rollout 64
python experiment2/train.py --steps 1000000 --workers 4
```

Training defaults to `--device cuda` and refuses silent CPU fallback when CUDA
is unavailable. PPO inference, gradients and optimizer tensors use the NVIDIA
GPU; four independent CPU MuJoCo workers collect experience for one shared PPO
policy. This is parallel rollout collection, not GPU physics or four independent
policies. Rendering also uses the GPU. Small MLP updates and CPU physics/video
work can leave GPU utilization low; CUDA alone does not guarantee a speedup.

`device.json` records the selected GPU and starting checkpoint timestep.
`runtime.json` records actual policy/optimizer devices and live timesteps.
New checkpoints include normalization every 2,048 steps. Resuming retains model,
optimizer, observation statistics and timestep count, while `--rollout` sets the
new rollout length. Environment episodes restart; this is not an exact replay
of every worker's in-flight physics or random-number state.

To stop gracefully, create an empty `STOP` file inside the active run directory;
the callback finishes the current vector step and saves `policy.zip` and
`normalize.pkl`. The CPU option remains available as `--device cpu`.

`resume_gpu.py --after-run runs/RUN` waits for an existing run's final checkpoint
and launches a CUDA continuation without terminating or discarding that run.

Each run has a unique directory under `experiment2/runs`. Every completed
training/evaluation throw saves `swing.mp4`, full physics states and outcome
metrics in `episodes/worker_*/`. Videos preserve failures as well as successes.
Rendering every throw reduces training throughput. Interrupted, unfinished
episodes are not presented as completed throws.

`policy.zip` and `normalize.pkl` must stay together. Resume or evaluate with:

```powershell
python experiment2/train.py --resume experiment2/runs/RUN/policy.zip --steps 1000000 --workers 4
python experiment2/train.py --evaluate experiment2/runs/RUN/policy.zip --episodes 100 --seed 73129
python experiment2/train.py --resume experiment2/runs/RUN/policy.zip --steps 1000000 --randomize
python experiment2/train.py --evaluate experiment2/runs/RUN/policy.zip --episodes 100 --seed 93129 --randomize
```

Open the live checkpoint switcher with `python experiment2/play.py`. Choose the
best evaluated policy or any saved PPO checkpoint, then load it, run, pause and
reset throws in the MuJoCo room. Use **Refresh checkpoints** while training is
active; completed policy/normalization pairs become selectable together. The
best-evaluated entry shows its measured evaluation count and bullseyes. A
training checkpoint is labeled by run and timestep rather than being called
best before evaluation.

Initial training uses the nominal fixed setup. After clean releases and board
hits improve, `--randomize` varies mass/inertia ±5%, hand friction ±15% and initial
dart placement ±0.1 mm. Evaluation disables observation-statistic updates and
uses deterministic policy actions. Fixed-scene repeats test reproducibility;
randomized held-out seeds test generalization. Report bullseye rate, board-hit
rate, miss distance, clean-release rate, failure types and actuator violations,
not just training reward. Multiple independent training seeds are needed before
claiming reliability. A short smoke run checks plumbing, not task convergence.

## Validation and current limits

`checks.json` includes Gym contract checks, repeatable reset checks, real-physics
rollouts and synthetic near-board fixtures testing the reward gate. The fixtures
place a dart near the board and are **not learned throws or performance evidence**.

Reference finger commands respect the source velocity limits, but externally
driven measured finger speeds can exceed those limits during rapid swings.
They are penalized and logged; this is not a hardware-qualified controller.
Contact/friction calibration, actuator dynamics, latency, balance and evaluation
with measured hardware parameters remain necessary for sim-to-real transfer.

The implementation follows [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
and [Gymnasium environment contracts](https://gymnasium.farama.org/introduction/create_custom_env/).
The installed smoke-test stack is PyTorch 2.6.0, SB3 2.8.0 and Gymnasium 1.2.3.
