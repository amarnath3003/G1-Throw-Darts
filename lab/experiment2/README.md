# Experiment 2 · Contact-based PPO throwing

The controller starts from a calibrated throwing motion. PPO learns bounded
corrections around that prior. A torque-limited G1 arm and Inspire hand release
a free dart, and MuJoCo evaluates flight and contact. Board hits require
valid release and tip-first contact. Inner bulls land within 6.35 mm of center.

## Held-out physical simulation

| Checkpoint | Board hits | Inner bulls | Robot collisions | Mean capped target error |
| --- | ---: | ---: | ---: | ---: |
| Start | 112/128 | 10/128 | 3 | 8.98 cm |
| Accuracy | 113/128 | 13/128 | 2 | 8.37 cm |
| Bullseye | 116/128 | 12/128 | 3 | 7.01 cm |
| Final | 120/128 | 5/128 | 4 | 5.97 cm |

The final checkpoint led on board hits and capped error. The accuracy checkpoint
recorded the most inner bulls in this comparison. The [evaluation files](analysis/extension_holdout_20261103/)
contain per-throw records and the aggregate comparison. The extension
checkpoints are archived separately from this repository.

## Reproduce the calibrated milestone

Install dependencies with `python -m pip install -r experiment2/requirements.txt`
and run `python experiment2/replay_board_hit.py --verify-only` from the `lab/` directory.
The [milestone bundle](milestones/first_board_hit/README.md) includes the policy,
normalizer, calibrated reference and recording. Its impact was 6.634 cm from
center. The [later 1.17 mm impact video](../progress_videos/12_ppo_bullseye_training_throw.mp4)
comes from an individual training rollout.

See [the reward design](REWARD_REVISION.md) for the earlier learning baseline.
Training requires a Linux training environment. Keep each checkpoint with its
matching normalizer and configuration and retain the per-swing video archive.
