"""Bounded SSH-only residual PPO pilot with physical evaluation and paired checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from env import HERE
from pickup_ppo_env import PickupResidualEnv, load_config


def write_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_episode(env, seed, model=None, normalizer=None):
    observation, reset_info = env.reset(seed=seed)
    states, timestamps, records = [env.data.qpos.copy()], [0.], []
    total_reward = 0.
    done = False
    while not done:
        if model is None:
            action = np.zeros(13)
        else:
            action, _ = model.predict(normalizer.normalize_obs(observation), deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        states.append(env.data.qpos.copy())
        timestamps.append(env.elapsed)
        records.append([env.elapsed, env.metrics['clearance'], env.hold, *env.metrics['forces']])
    return dict(seed=seed, **reset_info, **info, reward=total_reward), dict(
        qpos=np.asarray(states), time_s=np.asarray(timestamps),
        contact_log=np.asarray(records), contact_columns=np.array(['time_s', 'clearance_m', 'hold_s', 'thumb_N', 'index_N', 'middle_N']))


def evaluate(config, directory, model=None, normalizer=None, fine=True):
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    env = PickupResidualEnv(config, randomize=True)
    # Identical small selection set at every evaluation. Final test seeds stay untouched.
    for index in range(config['eval_episodes']):
        env.randomize = index != 0
        result, trajectory = run_episode(env, config['eval_seed']+index, model, normalizer)
        result.update(timestep_s=config['physics_dt'], nominal=index == 0)
        records.append(result)
        np.savez_compressed(directory / f'episode_{index:02d}.npz', **trajectory)
        print(json.dumps(dict(evaluation=directory.name, **result)), flush=True)
    env.close()
    if fine:
        env = PickupResidualEnv(config, randomize=False, timestep=.00025)
        result, trajectory = run_episode(env, config['eval_seed'], model, normalizer)
        result.update(timestep_s=.00025, nominal=True)
        records.append(result)
        np.savez_compressed(directory / 'episode_fine.npz', **trajectory)
        print(json.dumps(dict(evaluation=directory.name, **result)), flush=True)
        env.close()
    ordinary = records[:config['eval_episodes']]
    summary = dict(episodes=records, success_rate=float(np.mean([r['is_success'] for r in ordinary])),
                   mean_hold_s=float(np.mean([r['max_hold_s'] for r in ordinary])),
                   mean_lift_m=float(np.mean([r['max_lift_m'] for r in ordinary])),
                   fine_success=records[-1]['is_success'] if fine else None,
                   selection_evaluation=True, independent_test=False)
    write_json(directory / 'metrics.json', summary)
    return summary


def bundle(model, normalizer, config, run, step):
    destination = run / f'checkpoints/step_{step:09d}'
    temporary = destination.with_name(destination.name + '.pending')
    temporary.mkdir(parents=True, exist_ok=False)
    model.save(temporary / 'policy.zip')
    normalizer.save(temporary / 'normalizer.pkl')
    write_json(temporary / 'config.json', config)
    shutil.copy2(run / 'provenance.json', temporary / 'provenance.json')
    write_json(temporary / 'manifest.json', dict(transitions=step,
               policy_sha256=digest(temporary / 'policy.zip'),
               normalizer_sha256=digest(temporary / 'normalizer.pkl'),
               provenance_sha256=digest(temporary / 'provenance.json')))
    temporary.rename(destination)
    return destination


class PickupProgress(BaseCallback):
    def __init__(self, config, run):
        super().__init__()
        self.config, self.run = config, run
        self.started = time.monotonic()
        self.episodes = []
        self.next_eval = config['eval_every']
        self.last_saved = -1
        baseline = json.loads((run / 'baseline/metrics.json').read_text())
        self.best_score = (baseline['success_rate'], baseline['mean_hold_s'], baseline['mean_lift_m'])
        self.stale = 0
        self.stop_reason = None
        self.latest_evaluation = None

    def status(self, state='training', **extra):
        recent = self.episodes[-40:]
        write_json(self.run / 'status.json', dict(
            state=state, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(),
            transitions=self.num_timesteps, target_transitions=self.config['total_transitions'],
            elapsed_seconds=time.monotonic()-self.started, episodes=len(self.episodes),
            recent_training_success_rate=float(np.mean([r['is_success'] for r in recent])) if recent else None,
            recent_training_mean_lift_m=float(np.mean([r['max_lift_m'] for r in recent])) if recent else None,
            recent_training_mean_hold_s=float(np.mean([r['max_hold_s'] for r in recent])) if recent else None,
            latest_evaluation=self.latest_evaluation, stop_reason=self.stop_reason, **extra))

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                record = {key: info[key] for key in ('is_success', 'reason', 'max_lift_m', 'max_hold_s', 'elapsed_s')}
                record['transitions'] = self.num_timesteps
                self.episodes.append(record)
                with (self.run / 'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        if self.n_calls % 50 == 0:
            self.status()
        if (self.run / 'STOP').exists():
            self.stop_reason = 'user_stop_file'
        return self.stop_reason is None

    def _on_rollout_start(self):
        # Called after the preceding PPO update. Saves policy and current moments together.
        self.save_and_evaluate()

    def save_and_evaluate(self, final=False):
        if self.num_timesteps == self.last_saved:
            return
        checkpoint = bundle(self.model, self.training_env, self.config, self.run, self.num_timesteps)
        self.last_saved = self.num_timesteps
        if self.num_timesteps == 0:
            write_json(self.run / 'best.json', dict(checkpoint=str(checkpoint.relative_to(self.run)),
                       transitions=0, score=self.best_score, baseline=True))
        write_json(self.run / 'latest.json', dict(checkpoint=str(checkpoint.relative_to(self.run)), transitions=self.num_timesteps))
        if self.num_timesteps >= self.next_eval or final:
            self.status('evaluating')
            metrics = evaluate(self.config, checkpoint / 'evaluation', self.model, self.training_env)
            self.latest_evaluation = dict(transitions=self.num_timesteps, **metrics)
            score = (metrics['success_rate'], metrics['mean_hold_s'], metrics['mean_lift_m'])
            if score > self.best_score:
                self.best_score = score
                self.stale = 0
                write_json(self.run / 'best.json', dict(checkpoint=str(checkpoint.relative_to(self.run)),
                           transitions=self.num_timesteps, score=score))
            else:
                self.stale += 1
            self.next_eval = (self.num_timesteps//self.config['eval_every']+1)*self.config['eval_every']
            if self.stale >= 5:
                self.stop_reason = 'five_evaluations_without_physical_improvement'
        self.status()


def provenance(config, run):
    source_dir = run / 'sources'
    source_dir.mkdir()
    files = [HERE / name for name in ('train_pickup.py', 'pickup_ppo_env.py', 'try_pickup.py',
                                      'env.py', 'retarget_arm.py', 'retarget_pickup.py')]
    files += [HERE.parent / 'dart_model.py', HERE.parent / 'dart_profile.json']
    for path in files:
        shutil.copy2(path, source_dir / path.name)
    reference = HERE / config['reference']
    reference_dir = run / 'reference'
    reference_dir.mkdir()
    for name in ('commands.npz', 'reset.json', 'report.json', 'video_guidance.json', 'video_guidance.npz', 'provenance.json'):
        shutil.copy2(reference / name, reference_dir / name)
    config_dir = run / 'physics'
    config_dir.mkdir()
    for name in ('scene.json', 'scene.mjb', 'grasp_target.json'):
        shutil.copy2(HERE / 'configs' / name, config_dir / name)
    hashes = {p.relative_to(run).as_posix(): digest(p) for p in run.rglob('*') if p.is_file()}
    write_json(run / 'provenance.json', dict(files_sha256=hashes, python=sys.version,
               mujoco=mujoco.__version__, stable_baselines3=stable_baselines3.__version__,
               torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
               input_scope='Clip 4276 closure and simulated geometry; privileged simulator observations',
               test_seeds_used=False, device='cpu'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'train', 'render'])
    parser.add_argument('--config', type=Path, default=HERE / 'configs/pickup_ppo.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--trajectory', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(1)
    if args.mode == 'render':
        from try_pickup import render_trial
        env = PickupResidualEnv(config, randomize=False)
        trajectory = np.load(args.trajectory)
        # Re-sample 50 Hz stored physics poses to the renderer's 30 Hz playback.
        times = np.arange(0., trajectory['time_s'][-1]+1e-6, 1/30)
        indices = np.searchsorted(trajectory['time_s'], times).clip(0, len(trajectory['time_s'])-1)
        args.out.mkdir(parents=True, exist_ok=True)
        success = bool(np.max(trajectory['contact_log'][:, 2]) >= config['success_hold_s']-1e-9)
        render_trial(env.scene, args.out, trajectory['qpos'][indices], times, success,
                     controller_label='Residual PPO pickup evaluation')
        return
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'config.json', config)
    if args.mode == 'check':
        env = PickupResidualEnv(config, randomize=False)
        check_env(env, warn=True)
        # Invalid actions and unsupported placements must fail explicitly.
        try:
            env.step(np.full(13, np.nan))
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite action was accepted')
        env.reset_reference['dart_xy'] = [0., 0.]
        try:
            env.reset(seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported reset was accepted')
        env.close()
        results = evaluate(config, args.out / 'zero_residual')
        if not results['episodes'][0]['is_success']:
            raise AssertionError('Zero residual did not reproduce nominal acquisition')
        provenance(config, args.out)
        write_json(args.out / 'checks.json', dict(passed=True, gymnasium=True, nonfinite_rejected=True,
                   unsupported_reset_rejected=True, zero_residual_nominal_success=True,
                   fine_timestep_success=results['fine_success'], config_sha256=digest(args.config),
                   source_sha256={name: digest(HERE / name) for name in ('train_pickup.py', 'pickup_ppo_env.py', 'try_pickup.py')}))
        return
    if platform.system() != 'Linux' or not str(HERE).startswith('/home/ubuntu/darts-ppo/'):
        raise RuntimeError('New training is authorized only on the project SSH server')
    if args.preflight is None:
        raise ValueError('Training requires the physical preflight directory')
    checks = json.loads((args.preflight / 'checks.json').read_text())
    if not checks['passed'] or checks['config_sha256'] != digest(args.config):
        raise ValueError('Preflight config mismatch')
    for name, expected in checks['source_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError('Preflight source mismatch: ' + name)
    provenance(config, args.out)
    shutil.copytree(args.preflight / 'zero_residual', args.out / 'baseline')
    shutil.copy2(args.preflight / 'checks.json', args.out / 'preflight.json')
    workers = SubprocVecEnv([lambda: Monitor(PickupResidualEnv(config)) for _ in range(config['workers'])], start_method='spawn')
    workers.seed(config['train_seed'])
    normalizer = VecNormalize(workers, norm_obs=True, norm_reward=False, clip_obs=10.)
    model = PPO('MlpPolicy', normalizer, learning_rate=config['learning_rate'],
                n_steps=config['rollout_steps'], batch_size=config['batch_size'], n_epochs=config['epochs'],
                gamma=config['gamma'], gae_lambda=config['gae_lambda'], clip_range=config['clip_range'],
                target_kl=config['target_kl'], ent_coef=0., device='cpu', seed=config['train_seed'],
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=config['initial_log_std']), verbose=1)
    # Deterministic initial policy exactly matches zero residual; exploration is small.
    torch.nn.init.zeros_(model.policy.action_net.weight)
    torch.nn.init.zeros_(model.policy.action_net.bias)
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(args.out / 'logs'), ['stdout', 'csv']))
    progress = PickupProgress(config, args.out)
    try:
        model.learn(total_timesteps=config['total_transitions'], callback=progress)
        progress.save_and_evaluate(final=True)
        progress.status('stopped' if progress.stop_reason else 'pilot_complete')
    except Exception:
        write_json(args.out / 'error.json', dict(traceback=traceback.format_exc()))
        progress.status('failed')
        raise
    finally:
        normalizer.close()


if __name__ == '__main__':
    main()
