"""Read-only training audit and matched-seed checkpoint evaluation with videos."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from audit import evaluate

HERE=Path(__file__).resolve().parent


def metrics(rows):
    def median(values):
        values=[v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(values)) if values else None
    rel=[r for r in rows if r.get('release')]
    valid=[r for r in rows if r.get('release') and r['outcome'] in ('environment_hit','board','timeout')]
    trajectory=lambda key:median([(r.get('trajectory') or {}).get(key) for r in rel])
    return {'episodes':len(rows),'outcomes':dict(Counter(r['outcome'] for r in rows)),
            'board_hits':sum(r['outcome']=='board' for r in rows),
            'bullseyes':sum(bool(r['is_success']) for r in rows),
            'clean_flight_rate':len(valid)/len(rows) if rows else None,
            'median_velocity_gap_m_s':trajectory('velocity_gap_m_s'),
            'median_vertical_deficit_m_s':trajectory('vertical_velocity_deficit_m_s'),
            'median_predicted_board_error_m':trajectory('predicted_board_error_m'),
            'median_closest_tip_distance_m':median([r.get('closest_tip_distance_m') for r in valid]),
            'median_peak_height_m':median([r.get('peak_height_m') for r in valid]),
            'median_vx_m_s':median([r['release']['velocity'][0] for r in valid]),
            'median_vz_m_s':median([r['release']['velocity'][2] for r in valid]),
            'median_required_vz_m_s':trajectory('required_vz_at_current_vx_m_s'),
            'mean_return':float(np.mean([sum(r['reward_terms'].values()) for r in rows])) if rows else None}


def plot(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    batches=report['training']['batches'];x=[r['ending_episode'] for r in batches]
    axes[0].plot(x,[r['median_closest_tip_distance_m'] for r in batches],marker='o',label='Actual closest approach')
    axes[0].set(xlabel='Completed training throws',ylabel='Tip-to-bull distance (m)',title='Physical approach: lower is better')
    axes[1].plot(x,[r['median_vz_m_s'] for r in batches],marker='o',label='Actual upward speed')
    axes[1].plot(x,[r['median_required_vz_m_s'] for r in batches],ls='--',label='Ballistic speed needed')
    axes[1].set(xlabel='Completed training throws',ylabel='Upward release speed (m/s)',title='The upward-speed deficit remains')
    rows=report['evaluations'];labels=[r['label'] for r in rows]
    axes[2].bar(range(len(rows)),[r['metrics']['median_closest_tip_distance_m'] or 0 for r in rows],color=['#777777','#e67e22','#2980b9','#239b56'])
    axes[2].set_xticks(range(len(rows)),labels,rotation=15,ha='right',fontsize=8)
    axes[2].set(ylabel='Median actual closest approach (m)',title='Matched deterministic checkpoints')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    for ax in axes[:2]:ax.legend(fontsize=8)
    fig.savefig(root/'progress.png',dpi=160)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'runs'/'20261029T155558346875')
    p.add_argument('--episodes',type=int,default=8);a=p.parse_args();torch.set_num_threads(1)
    root=HERE/'analysis'/('progress_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'));root.mkdir(parents=True)
    # Snapshot older completed files/checkpoints; never interfere with the trainer.
    cutoff=time.time()-10
    rows=[]
    for f in sorted((a.run/'episodes').glob('*/*/result.json'),key=lambda f:f.stat().st_mtime):
        if f.stat().st_mtime>cutoff:continue
        try:rows.append(json.loads(f.read_text()))
        except json.JSONDecodeError:continue
    batches=[]
    for start in range(0,len(rows)-127,128):
        batches.append(dict(ending_episode=start+128,**metrics(rows[start:start+128])))
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'run':str(a.run),
            'runtime_snapshot':json.loads((a.run/'runtime.json').read_text()),
            'training':{'total':metrics(rows),'first200':metrics(rows[:200]),'last200':metrics(rows[-200:]),'batches':batches},
            'evaluations':[]}
    (root/'progress.json').write_text(json.dumps(report,indent=2))
    print('OUTPUT '+str(root),flush=True)
    print('TRAINING '+json.dumps({k:report['training'][k] for k in ('first200','last200')}),flush=True)
    cps=sorted([f for f in (a.run/'checkpoints').glob('rl_model_*_steps.zip') if f.stat().st_mtime<cutoff],key=lambda f:int(f.stem.split('_')[-2]))
    if not cps:raise RuntimeError('No stable checkpoint found')
    first=cps[0];last=cps[-1];mid=min(cps,key=lambda f:abs(int(f.stem.split('_')[-2])-(int(first.stem.split('_')[-2])+int(last.stem.split('_')[-2]))/2))
    old=HERE/'runs'/'20261029T144134591936'/'policy.zip'
    candidates=[('Original 200k',old,'legacy'),('Early revised',first,'arc_v2'),('Mid revised',mid,'arc_v2'),('Latest revised',last,'arc_v2')]
    seeds=list(range(74101,74101+a.episodes))
    for i,(label,model,variant) in enumerate(candidates):
        print('EVALUATING '+label+' '+str(model),flush=True)
        evaluation=evaluate(model,variant,root/f'eval_{i}',seeds,True)
        item={'label':label,'model':str(model),'metrics':metrics(evaluation['episodes']),'details':evaluation}
        report['evaluations'].append(item)
        (root/'progress.json').write_text(json.dumps(report,indent=2))
        print('RESULT '+json.dumps({'label':label,**item['metrics']}),flush=True)
    plot(report,root)
    print('DONE '+str(root),flush=True)


if __name__=='__main__':main()
