"""Seeded real-physics evaluation, action/actuation telemetry and every video."""
import runtime_bootstrap
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import cloudpickle
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from env import build_env
from train import Archive,normalizer_for
from progress_check import metrics
from bounded_policy import BoundedActorCriticPolicy
from throw_diagnostics import diagnostics


def evaluate(model,variant,root,seeds,randomize=True,bounded_actions=False,throw_profile=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    options={'custom_objects':{'policy_class':BoundedActorCriticPolicy}} if bounded_actions else {}
    policy=PPO.load(str(model),device='cuda',**options)
    with normalizer_for(model).open('rb') as f:norm=cloudpickle.load(f)
    env=build_env(variant=variant,throw_profile=throw_profile,randomize=randomize,record=True)
    archive=Archive(env,root/'episodes');rows=[];clip=[];slew=[]
    try:
        for seed in seeds:
            obs,_=archive.reset(seed=seed);done=False;traces=[]
            while not done:
                normalized=norm.normalize_obs(obs[None,:])
                with torch.no_grad():
                    tensor,_=policy.policy.obs_to_tensor(normalized)
                    distribution=policy.policy.get_distribution(tensor)
                    mean=distribution.distribution.mean.cpu().numpy()[0]
                    mode=distribution.mode().cpu().numpy()[0]
                action,_=policy.predict(normalized,deterministic=True);action=action[0]
                desired=action[:9]*env.scale if len(action)==10 else None
                if len(action)!=10:limited=np.zeros(9,dtype=bool)
                elif env.rate_control:limited=np.abs(env.residual+action[:9]*env.slew_limit())>env.scale+1e-8
                else:limited=np.abs(desired-env.residual)>np.asarray(env.slew_limit())+1e-8
                if not env.released:clip.append(np.abs(mode)>1);slew.append(limited)
                traces.append({'time_s':env.t,'mean':mean.tolist(),'mode':mode.tolist(),'action':action.tolist(),
                    'residual':env.residual.tolist(),'slew_limited':limited.tolist(),
                    'q':env.d.qpos[env.qadr].tolist(),'qd':env.d.qvel[env.vadr].tolist(),
                    'dart_com':env.d.xipos[env.sim.dart].tolist(),'velocity':env.velocity().tolist(),
                    'commanded':env.commanded,'released':env.released})
                obs,_,term,trunc,info=archive.step(action);done=term or trunc
            row=dict(info,seed=seed);rows.append(row)
            folder=archive.folder/f'{archive.number-1:06d}'
            (folder/'actions.json').write_text(json.dumps(traces,indent=2))
            print('EPISODE '+json.dumps({'seed':seed,'outcome':row['outcome'],'closest':row['closest_tip_distance_m'],
                'release':row['release']}),flush=True)
    finally:archive.close()
    report={'model':str(model),'variant':variant,'throw_profile':throw_profile,'seeds':seeds,'randomize':randomize,'metrics':metrics(rows),
            'policy_class':type(policy.policy).__name__,'action_distribution':type(policy.policy.action_dist).__name__,
            'diagnostics':diagnostics(rows),
            'control_mode':'episode_parameters' if (throw_profile or {}).get('controller')=='parameter_ppo_v1' else 'residual_rate' if variant in ('arc_v6','arc_v7','arc_v8','arc_v9','arc_v10') else 'residual_position',
            'pre_release_clipped_fraction_by_action':np.mean(clip,axis=0).tolist() if clip else None,
            'pre_release_slew_limited_fraction_by_joint':np.mean(slew,axis=0).tolist() if slew else None,
            'episodes':rows,'created_utc':datetime.now(timezone.utc).isoformat()}
    (root/'evaluation.json').write_text(json.dumps(report,indent=2))
    print('SUMMARY '+json.dumps(report['metrics']),flush=True)
    print('OUTPUT '+str(root),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--variant',default=None);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=8);p.add_argument('--seed',type=int,default=74101)
    p.add_argument('--bounded-actions',action='store_true')
    p.add_argument('--throw-profile',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    run=a.model.parent.parent if a.model.parent.name=='checkpoints' else a.model.parent
    config=json.loads((run/'config.json').read_text())
    variant=a.variant or config.get('variant') or 'legacy'
    profile=json.loads(a.throw_profile.read_text()) if a.throw_profile else config.get('throw_profile')
    if profile is not None and 'profile' in profile:profile=profile['profile']
    evaluate(a.model,variant,a.output,list(range(a.seed,a.seed+a.episodes)),bounded_actions=a.bounded_actions,throw_profile=profile)


if __name__=='__main__':main()
