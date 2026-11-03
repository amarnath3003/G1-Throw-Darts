# Calibrated board hit — 30 October 2026

Trial 91, seed 81001, produced a clean release and valid tip-first board impact
**6.634 cm from the bullseye**. This is not an inner bullseye (radius 0.635 cm)
or a held-out success-rate result. Joint-limit violation was zero; measured
finger speeds still exceeded the reference command speed, so this controller
is not ready to deploy on hardware.

The frozen 403,564-step PPO policy is paired with its exact observation
normalizer. A simulator optimizer adjusted the arm reference, rate ramp and
waist aim. The dart remains free and contact-held; gravity, aerodynamics,
contact geometry, mass, friction and torque ceilings were not changed to aim.
After release, physics continues with fixed residual offsets and the normal
follow-through prior. No projectile steering or launch impulse is used.

This bundle includes the policy, normalizer, calibrated profile, original
result, physics states, control archive and video. Search trial provenance and
checksums are in `provenance.json`. The calibration is additional optimization
around a frozen PPO policy, not a claim that PPO alone learned the board hit.

From `hackathon` with the project Python environment:

```powershell
python experiment2/replay_board_hit.py
```

The script first simulates a fresh throw using the saved policy/profile/seed,
checks its physical hit against the saved impact error, and saves a new video
under `experiment2/runs/board_hit_replay_*`. It then opens an interactive MuJoCo
replay of those verified states at half speed. SPACE pauses, R restarts, and V
switches between room and board cameras. Mouse controls can orbit and zoom.
The last contact frame is held before the throw repeats. Board penetration and
sticking are not modeled. Use `--verify-only` to reproduce without opening a window.

The broad/fine measured-target reward and terminal-flight credit are included
for subsequent PPO work. The saved trial itself uses a frozen policy; follow-on
learning and independent validation remain separate milestones.
