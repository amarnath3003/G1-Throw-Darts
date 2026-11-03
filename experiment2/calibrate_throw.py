"""Optimize joint references with actual MuJoCo outcomes; archive every swing.

The policy stays frozen. Search seeds are distinct from final validation.
This is explicitly simulator calibration, not PPO interaction-step credit.
"""
import runtime_bootstrap  # before optional native dependencies pulled by SB3
import argparse
import json
from pathlib import Path
import shutil
import cloudpickle
import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from stable_baselines3 import PPO
from calibrated_env import CalibratedThrowEnv
from train import Archive, normalizer_for


FIELDS = ['elbow_back','shoulder_finish','rate_scale','roll_finish','elbow_finish','aim_yaw']
BOUNDS = [(-.275,.10),(-.55,.8),(.8,1.9),(-.4,.5),(0.,.8),(-.3,.15)]


def actual_cost(info):
    if info.get('release') is None or info['outcome'] not in ('environment_hit','board','timeout'):
        return 5.
    crossing = info.get('board_plane_crossing')
    if info.get('reward_terms',{}).get('board_impact',0)>0:
        return float(info['impact_error_m'])
    if info['outcome']=='board':
        return 2.+float(info['impact_error_m'])
    if crossing:
        return float(crossing['error_m'])+.05
    return 2.+float(info.get('closest_tip_distance_m') or 3.)


def main():
    global FIELDS,BOUNDS
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generations',type=int,default=8)
    p.add_argument('--seed',type=int,default=81001)
    p.add_argument('--seeds',type=int,default=1)
    p.add_argument('--profile',type=Path)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--local',action='store_true')
    p.add_argument('--aim-only',action='store_true',help='Keep the established stroke; refine shoulder elevation and torso aim')
    args=p.parse_args();torch.set_num_threads(1)
    if args.aim_only:
        FIELDS=['shoulder_finish','aim_yaw'];BOUNDS=[(-.55,.8),(-.3,.15)]
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    policy=PPO.load(str(args.model),device='cuda')
    with normalizer_for(args.model).open('rb') as f:norm=cloudpickle.load(f)
    env=CalibratedThrowEnv(randomize=True,record=True)
    archive=Archive(env,root/'episodes')
    records=[];best=np.inf
    start=json.loads(args.profile.read_text()) if args.profile else {}
    if 'profile' in start:start=start['profile']
    (root/'config.json').write_text(json.dumps(dict(model=str(args.model),search_seed=args.seed,
        seeds=args.seeds,fields=FIELDS,bounds=BOUNDS,profile=start,method='frozen PPO joint-reference calibration'),indent=2))
    for source in ('calibrated_env.py','calibrate_throw.py','env.py','trajectory.py'):
        shutil.copy2(Path(__file__).parent/source,root/source)

    def evaluate(x):
        nonlocal best
        if (root/'STOP').exists():raise KeyboardInterrupt('Graceful calibration stop')
        profile=dict(start,**dict(zip(FIELDS,map(float,x))));rows=[]
        try:env.configure(profile)
        except ValueError as error:
            return 10.
        for seed in range(args.seed,args.seed+args.seeds):
            obs,_=archive.reset(seed=seed);done=False
            while not done:
                action,_=policy.predict(norm.normalize_obs(obs[None,:]),deterministic=True)
                obs,_,term,trunc,info=archive.step(action[0]);done=term or trunc
            rows.append(dict(info,seed=seed,video=str(archive.folder/f'{archive.number-1:06d}'/'swing.mp4')))
        cost=float(np.mean([actual_cost(r) for r in rows]))
        record=dict(trial=len(records),cost=cost,profile=profile,episodes=rows)
        records.append(record)
        with (root/'trials.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if cost<best:
            best=cost;(root/'best.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(trial=record['trial'],cost=cost,best=best,
            outcomes=[r['outcome'] for r in rows],profile=profile)),flush=True)
        return cost

    try:
        x0=np.array([start.get(k,env.profile[k]) for k in FIELDS])
        evaluate(x0)
        if not args.evaluate_only:
            if args.local:
                result=minimize(evaluate,x0,method='Nelder-Mead',bounds=BOUNDS,
                    options=dict(maxfev=180,xatol=.0002,fatol=.0002,
                                 initial_simplex=np.vstack([x0,*[x0+np.eye(len(FIELDS))[i]*.015 for i in range(len(FIELDS))]])))
            else:
                result=differential_evolution(evaluate,BOUNDS,x0=x0,popsize=6,
                    maxiter=args.generations,seed=8321,polish=False,tol=.002,atol=.002)
            (root/'optimizer.json').write_text(json.dumps(dict(x=result.x.tolist(),fun=float(result.fun),
                message=str(result.message),nfev=int(result.nfev)),indent=2))
    except KeyboardInterrupt:
        print('Calibration stopped after saving all completed throws.',flush=True)
    finally:
        archive.close()
        (root/'summary.json').write_text(json.dumps(dict(completed_trials=len(records),best_cost=best,
            completed_swings=archive.number,policy_device=str(policy.device)),indent=2))


if __name__=='__main__':main()
