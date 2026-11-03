"""Reproduce the saved calibrated hit, then replay its real MuJoCo states."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from calibrated_env import CalibratedThrowEnv

HERE = Path(__file__).resolve().parent


def reproduce(bundle):
    import cloudpickle
    import torch
    from stable_baselines3 import PPO
    from train import Archive

    torch.set_num_threads(1)
    metadata = json.loads((bundle / 'provenance.json').read_text())
    profile = json.loads((bundle / 'profile.json').read_text())
    expected = json.loads((bundle / 'result.json').read_text())
    env = CalibratedThrowEnv(profile=profile, randomize=True, record=True)
    policy = PPO.load(str(bundle / 'policy.zip'), device='cpu')
    with (bundle / 'normalize.pkl').open('rb') as handle:
        normalizer = cloudpickle.load(handle)
    output = HERE / 'runs' / ('board_hit_replay_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    archive = Archive(env, output / 'episodes')
    try:
        obs, _ = archive.reset(seed=metadata['seed'])
        done = False
        while not done:
            action, _ = policy.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, terminated, truncated, info = archive.step(action[0])
            done = terminated or truncated
        valid = info['outcome'] == 'board' and info['reward_terms'].get('board_impact', 0) > 0
        if not valid:
            raise RuntimeError(f'Saved hit did not reproduce: {info}')
        np.testing.assert_allclose(info['impact_error_m'], expected['impact_error_m'], atol=1e-6, rtol=0)
        (output / 'reproduction.json').write_text(json.dumps({
            'verified': True, 'seed': metadata['seed'], 'source_bundle': str(bundle),
            'outcome': info['outcome'], 'impact_error_m': info['impact_error_m'],
            'bullseye': bool(info['is_success']), 'method': 'frozen PPO plus calibrated joint references',
        }, indent=2))
        print('VERIFIED ' + str(output), flush=True)
        return output / 'episodes' / '000000'
    finally:
        archive.close()


def replay(bundle, recording):
    metadata = json.loads((bundle / 'provenance.json').read_text())
    info = json.loads((recording / 'result.json').read_text())
    profile = json.loads((bundle / 'profile.json').read_text())
    env = CalibratedThrowEnv(profile=profile, randomize=True)
    env.reset(seed=metadata['seed'])
    states = np.load(recording / 'states.npz')
    controls = dict(paused=False, restart=False, board=False, camera=True)

    def keypress(key):
        if key == 32:
            controls['paused'] = not controls['paused']
        elif key in (ord('R'), ord('r')):
            controls['restart'] = True
        elif key in (ord('V'), ord('v')):
            controls['board'] = not controls['board']
            controls['camera'] = True

    index = 0
    last = 0.
    try:
        with mujoco.viewer.launch_passive(env.m, env.d, key_callback=keypress,
                                         show_left_ui=False, show_right_ui=False) as viewer:
            viewer.opt.geomgroup[3] = 0
            while viewer.is_running():
                if controls['restart']:
                    index = 0
                    controls['restart'] = False
                if controls['camera']:
                    if controls['board']:
                        viewer.cam.lookat[:] = [3., 0., 1.73]
                        viewer.cam.distance = 1.0
                        viewer.cam.azimuth = 180
                        viewer.cam.elevation = 0
                    else:
                        viewer.cam.lookat[:] = [1.5, 0., 1.35]
                        viewer.cam.distance = 3.8
                        viewer.cam.azimuth = 90
                        viewer.cam.elevation = -8
                    controls['camera'] = False
                with viewer.lock():
                    env.d.qpos[:] = states['qpos'][index]
                    env.d.qvel[:] = states['qvel'][index]
                    env.d.time = float(states['time'][index])
                    mujoco.mj_forward(env.m, env.d)
                viewer.set_texts((None, None,
                    'VERIFIED TIP-FIRST BOARD HIT\nFrozen PPO + joint-reference calibration\n'
                    f'Impact error: {100*info["impact_error_m"]:.2f} cm; bullseye: {info["is_success"]}\n'
                    'Recorded physics replay at half speed\nSPACE pause | R restart | V board / room',
                    f'{states["time"][index]:.3f} s'))
                viewer.sync()
                now = time.perf_counter()
                if not controls['paused'] and now - last >= (.04 if index < len(states['time'])-1 else 3.):
                    index = (index + 1) % len(states['time'])
                    last = now
                time.sleep(.01)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=HERE / 'milestones' / 'first_board_hit')
    parser.add_argument('--recording', type=Path, help='Replay an already reproduced episode')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    recording = args.recording or reproduce(args.bundle)
    if not args.verify_only:
        replay(args.bundle, recording)


if __name__ == '__main__':
    main()
