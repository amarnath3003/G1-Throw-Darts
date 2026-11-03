"""Archive zero-residual and bounded-correction physics before launching PPO."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiment2'))
import runtime_bootstrap
import numpy as np
import torch
from train import Archive
from env import build_env, impact_reward
from throw_diagnostics import diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    profile = json.loads(args.profile.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    env = Archive(build_env(throw_profile=profile, randomize=not args.quick, record=True), args.output/'episodes')
    tests = [('zero', 98101+i, np.zeros(8)) for i in range(1 if args.quick else 8)]
    if not args.quick:
        for axis in range(8):
            for sign in (-1., 1.):
                action = np.zeros(8)
                action[axis] = sign*.5
                tests.append((f'axis_{axis}_{sign:g}', 98201, action))
        rng = np.random.default_rng(98301)
        tests.extend(('initial_exploration', 98301+i, np.tanh(rng.normal(0., .10, 8))) for i in range(8))
    rows = []
    started = time.monotonic()
    try:
        for label, seed, action in tests:
            obs, _ = env.reset(seed=seed)
            assert obs.shape == (65,) and np.isfinite(obs).all()
            _, reward, term, trunc, info = env.step(action)
            assert term and not trunc
            assert abs(sum(info['reward_terms'].values())-reward) < 1e-6
            assert abs(sum(row[2] for row in env.env.transition_controls)-reward) < 1e-6
            assert info['max_joint_limit_violation_rad'] <= .03
            assert not any(name in joint for name in env.env.names for joint in ('hip', 'knee', 'ankle', 'waist'))
            rows.append(dict(info, seed=seed, label=label))
            print(json.dumps(dict(label=label, outcome=info['outcome'], reward=reward,
                                  release=info['release'], swings=info['swing_checks'])), flush=True)
    finally:
        env.close()
    zeros = [row for row in rows if row['label']=='zero']
    assert all(row['release'] and all(check['passed'] for check in row['swing_checks']) for row in zeros)
    assert all(row['min_shoulder_lane_m'] >= 0 and row['min_dart_head_gap_m'] > .005 for row in zeros)
    assert all(row['outcome'] in ('environment_hit', 'board', 'timeout') for row in zeros)
    assert impact_reward(.00635, True, True)[1]
    assert not impact_reward(.00636, True, True)[1]
    assert not impact_reward(0., False, True)[1]
    assert not impact_reward(0., True, False)[1]
    report = dict(passed=True, controller=profile['controller'], reference_sha256=profile['reference_sha256'],
                  environment_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                      for name in ('experiment4/video_env.py','experiment2/env.py',
                                                   'inspire_grip.py','inspire_model.py','experiment2/parameter_env.py')},
                  elapsed_s=time.monotonic()-started, baseline=diagnostics(zeros),
                  all_trials=diagnostics(rows), episodes=rows)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print('PREFLIGHT PASSED '+json.dumps(report['baseline']), flush=True)


if __name__ == '__main__':
    main()
