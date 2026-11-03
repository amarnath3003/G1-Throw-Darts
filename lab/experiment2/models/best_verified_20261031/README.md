# Best verified system — 31 October 2026

Selected checkpoint: **404,588 PPO steps**, with a physics-calibrated joint reference.
The 405,100-step checkpoint tied its validation board-hit count but had worse
mean closest approach, so the earlier checkpoint was retained.

Final untouched test, seeds 84101–84132, deterministic policy actions and
randomized dart mass, hand friction and initial placement:

- **21/32 valid tip-first board hits (65.6%)**.
- **0/32 bullseyes**. The bullseye objective remains unmet.
- 26/32 clean flights; four robot hits, two hand recontacts, five other misses.
- Median impact error **among the 21 valid hits: 14.6 cm**; best: 12.5 cm.
- Every attempt has its original video, physics states, controls and result JSON.

This result supersedes small tuning demonstrations. In particular, the earlier
6.63 cm single-seed calibration throw is not the final test accuracy. Release
consistency under randomized conditions and systematic aiming error remain the
principal problems. No further model selection used the final test outcomes.

## Run or resume

From the hackathon repository directory, using the existing environment:

```powershell
..\video_to_g1\.venv\Scripts\python.exe experiment2/evaluate_diagnostic.py --model experiment2/models/best_verified_20261031/policy.zip --output experiment2/analysis/new_test --episodes 12 --seed 85101
```

The output directory must be new. The evaluator inherits the required saved
joint-reference profile and paired observation normalization automatically.

```powershell
..\video_to_g1\.venv\Scripts\python.exe experiment2/launch_training.py -- --resume experiment2/models/best_verified_20261031/policy.zip --device cuda --workers 4 --rollout 128 --steps 4096 --checkpoint-every 512 --randomize --eval-every 1024 --eval-episodes 12 --eval-seed 83101
```

Training is currently stopped. This command explicitly resumes CUDA PPO with
four CPU MuJoCo workers. All completed swings are recorded. Preserve policy,
normalization and config together. The bundle also contains source snapshots,
selection records, final evaluation, summary and policy hash.

The original left five-finger hand and free-contact dart physics are retained.
The robot pelvis remains supported. Hardware transfer has not been validated.

Full evidence: `experiment2/analysis/final_20261031/heldout/`.
Representative median-error valid hit, slowed four times:
`experiment2/analysis/final_20261031/representative_hit_4x_slow.mp4`.
Changes and calibration history: `experiment2/analysis/round400k/REPORT.md`.
