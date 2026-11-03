"""Read-only progress audit with matched-seed checkpoint evaluations."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from audit import evaluate
from train import normalizer_for

HERE=Path(__file__).resolve().parent


def metrics(rows):
    releases=[r for r in rows if r.get('release')]
    def median(values):
        values=[v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(values)) if values else None
    def trajectory(key):
        return median([(r.get('trajectory') or {}).get(key) for r in releases])
    return {
        'episodes':len(rows),'outcomes':dict(Counter(r['outcome'] for r in rows)),
        'bullseyes':sum(r['is_success'] for r in rows),
        'valid_board_hits':sum(r['outcome']=='board' and r['reward_terms'].get('board_impact',0)>0 for r in rows),
        'uninterrupted_forward_flight_rate':sum(r['outcome'] in ('environment_hit','board','timeout') and r.get('release') is not None for r in rows)/len(rows),
        'median_release_velocity_m_s':np.median([r['release']['velocity'] for r in releases],axis=0).tolist() if releases else None,
        'median_peak_height_m':median([r.get('peak_height_m') for r in rows]),
        'median_actual_closest_tip_distance_m':median([r.get('closest_tip_distance_m') for r in rows]),
        'median_velocity_gap_m_s':trajectory('velocity_gap_m_s'),
        'median_vertical_velocity_deficit_m_s':trajectory('vertical_velocity_deficit_m_s'),
        'median_predicted_board_error_m':trajectory('predicted_board_error_m'),
    }


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    run=args.run.resolve();root=HERE/'analysis'/('progress_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'));root.mkdir(parents=True)
    files=sorted((run/'episodes').glob('*/*/result.json'),key=lambda p:p.stat().st_mtime)
    rows=[];skipped=0
    for file in files:
        try:rows.append(json.loads(file.read_text()))
        except json.JSONDecodeError:skipped+=1 # Another worker may still be writing.
    n=len(rows);width=min(100,n)
    report={'run':str(run),'created_utc':datetime.now(timezone.utc).isoformat(),
            'training_snapshot':json.loads((run/'runtime.json').read_text()),
            'skipped_incomplete_results':skipped,'all_training':metrics(rows),
            'training_windows':{'first':metrics(rows[:width]),
                                'middle':metrics(rows[max(0,n//2-width//2):n//2+width//2]),
                                'latest':metrics(rows[-width:])},
            'matched_evaluation':{},'seeds':[73129,73130,73131,73132]}
    # Freeze the choice before evaluation; training continues independently.
    models=sorted((run/'checkpoints').glob('rl_model_*_steps.zip'),key=lambda p:int(p.stem.split('_')[2]))
    usable=[]
    for model in models:
        if not normalizer_for(model).exists():continue
        try:
            with zipfile.ZipFile(model) as archive:
                if archive.testzip() is None:usable.append(model)
        except zipfile.BadZipFile:continue
    selected={'early_calibrated':usable[0],'recent_calibrated':usable[-1],
              'original_200k':HERE/'runs'/'20261029T144134591936'/'policy.zip'}
    for name,model in selected.items():
        variant='legacy' if name=='original_200k' else json.loads((run/'config.json').read_text())['variant']
        print('EVALUATING '+name+' '+model.name,flush=True)
        result=evaluate(model,variant,root/name,report['seeds'],True)
        result['physical_metrics']=metrics(result['episodes'])
        result['checkpoint']=str(model);result['sha256']=hashlib.sha256(model.read_bytes()).hexdigest()
        report['matched_evaluation'][name]=result
        (root/'progress.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({name:result['physical_metrics']},indent=2),flush=True)
    (root/'progress.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    rolling=[metrics(rows[i:i+width]) for i in range(0,max(1,n-width+1),50)]
    xs=np.arange(len(rolling))*50+width
    for ax,key,title in zip(axes,['median_velocity_gap_m_s','median_actual_closest_tip_distance_m','uninterrupted_forward_flight_rate'],
                           ['Velocity gap to viable arc (m/s) ↓','Actual closest tip approach (m) ↓','Uninterrupted forward flight rate ↑']):
        ax.plot(xs,[r[key] for r in rolling],color='#2874a6');ax.set(title=title,xlabel='Completed training episode (100-episode windows)');ax.grid(alpha=.2)
    fig.savefig(root/'training_trend.png',dpi=150)
    print('REPORT '+str(root/'progress.json'),flush=True)


if __name__=='__main__':main()
