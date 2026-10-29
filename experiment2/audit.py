"""Audit existing real-physics throws, then compare checkpoint rollouts fairly."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import cloudpickle
import numpy as np
import torch
from stable_baselines3 import PPO
from env import DartThrowEnv
from train import Archive, normalizer_for
from trajectory import guidance

HERE=Path(__file__).resolve().parent


def summarize(rows):
    rel=[r for r in rows if r.get('release')]
    def median(key):
        vals=[r['trajectory'].get(key) for r in rel if r.get('trajectory') and r['trajectory'].get(key) is not None]
        return float(np.median(vals)) if vals else None
    return {'episodes':len(rows),'outcomes':dict(Counter(r['outcome'] for r in rows)),
            'bullseyes':sum(r['is_success'] for r in rows),
            'median_release_velocity_m_s':np.median([r['release']['velocity'] for r in rel],axis=0).tolist() if rel else None,
            'median_velocity_gap_m_s':median('velocity_gap_m_s'),
            'median_required_vz_m_s':median('required_vz_at_current_vx_m_s'),
            'median_vertical_deficit_m_s':median('vertical_velocity_deficit_m_s'),
            'median_minimum_ballistic_speed_m_s':median('minimum_ballistic_speed_m_s'),
            'median_predicted_board_error_m':median('predicted_board_error_m'),
            'median_old_prediction_reward':float(np.median([r.get('reward_terms',{}).get('release_prediction',0.) for r in rows]))}


def archived(run):
    files=sorted((run/'episodes').glob('*/*/result.json'),key=lambda f:f.stat().st_mtime)
    rows=[json.loads(f.read_text()) for f in files]
    for row in rows:
        if row.get('release'):
            r=row['release'];row['trajectory']=guidance(r['position'],r['velocity'],[3,0,1.73])
    return {'all':summarize(rows),'last400_chronological':summarize(rows[-400:])}


def evaluate(model,variant,folder,seeds,randomize):
    policy=PPO.load(str(model),device='cuda')
    with normalizer_for(model).open('rb') as f:norm=cloudpickle.load(f)
    env=Archive(DartThrowEnv(variant=variant,randomize=randomize,record=True),folder)
    rows=[]
    try:
        for seed in seeds:
            obs,_=env.reset(seed=seed);done=False
            while not done:
                action,_=policy.predict(norm.normalize_obs(obs[None,:]),deterministic=True)
                obs,reward,term,trunc,info=env.step(action[0]);done=term or trunc
            rows.append(info)
    finally:env.close()
    return {'summary':summarize(rows),'episodes':rows,'seeds':seeds,'randomize':randomize,'variant':variant}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    root=HERE/'analysis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');root.mkdir(parents=True)
    report={'archive':{},'matched_evaluation':{}}
    for name in ('20261029T132555290476','20261029T144134591936'):
        run=HERE/'runs'/name;report['archive'][name]=archived(run)
        if a.evaluate:
            for variant in ('legacy','arc_v2') if name=='20261029T144134591936' else ('legacy',):
                key=f'{name}_{variant}'
                report['matched_evaluation'][key]=evaluate(run/'policy.zip',variant,root/key,[73129,73130,73131,73132],True)
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    print('AUDIT '+str(root),flush=True)
    print(json.dumps({k:v['summary'] for k,v in report['matched_evaluation'].items()},indent=2),flush=True)


if __name__=='__main__':main()
