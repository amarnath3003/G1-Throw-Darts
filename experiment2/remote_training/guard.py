"""Independent remote guard; survives closing the local dashboard."""
import argparse,json,os,time
from pathlib import Path

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def assess(reports):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from throw_diagnostics import paired_error_interval
    best=None;best_report=None;bad=0;stale=0;history=[];reason=None;severe=0
    seen_steps=set();recent=[]
    start=min((r['timesteps'] for r in reports),default=0)
    for r in reports:
        s=r['summary'];why=[];interval=None
        distance=s.get('mean_capped_target_error_m',s['mean_penalized_closest_distance_m'])
        if best:
            if r.get('episodes') and best_report.get('episodes'):
                interval=paired_error_interval(best_report,r)
            if s['clean_flight_rate']<best['clean_flight_rate']-.09:
                why.append('clean flight rate down at least 9 points')
            if s['valid_board_hits']<=best['valid_board_hits']-max(3,int(.1*s['episodes'])):
                why.append('material board-hit regression')
            d=best.get('mean_capped_target_error_m',best['mean_penalized_closest_distance_m'])
            if distance>max(d*1.25,d+.03) and (interval is None or interval['lower']>.015):
                why.append('paired endpoint error regression (95% interval)')
        improved=best is None or tuple(s['selection_score'])>tuple(best['selection_score'])
        if improved:best=s;best_report=r
        # Reference and initial learner at the same step are one assessment,
        # not two failed opportunities to learn.
        is_update=not r.get('label') and r['timesteps']>start and r['timesteps'] not in seen_steps
        if is_update:
            seen_steps.add(r['timesteps']);recent.append(distance)
            bad=bad+1 if why else 0;stale=0 if improved else stale+1
            severe=severe+1 if s['clean_flight_rate']<.65 else 0
            recovering=len(recent)>=3 and recent[-1]<recent[-3]-.01
            if s.get('numerical_failure_count',0)>0:
                reason='Numerical instability or joint-limit failure; inspect immediately'
            elif severe>=2:
                reason='Severe reliability collapse in two evaluations (clean flights below 65%)'
            elif r['timesteps']-start>=1536 and bad>=6 and not recovering:
                reason='Six confirmed regressions without recovery; best checkpoint retained'
            elif stale>=24 and not recovering:
                reason='Twenty-four evaluations without improvement; review plateau'
        history.append(dict(step=r['timesteps'],**s,warnings=why,paired_error_interval=interval))
        if reason:break
    return dict(best=best,history=history,bad_streak=bad,stale_evaluations=stale,stop_reason=reason,
        guard_version=2,recovery_policy='warn first; 1536-step grace; six regressions; improving trend extends recovery')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args()
    root=a.run;config=read(root/'config.json',{});start=read(root/'device.json',{}).get('starting_timesteps',0)
    while True:
        reports=[]
        for path in sorted((root/'evaluations').glob('*/evaluation.json')):
            r=read(path,{})
            if r.get('summary') and not r.get('interrupted'):reports.append(r)
        reports.sort(key=lambda r:(r['timesteps'],0 if r.get('label') else 1))
        # Compare identical validation seeds only.
        if reports:reports=[r for r in reports if r.get('seeds')==reports[0].get('seeds')]
        status=assess(reports);runtime=read(root/'runtime.json',{});summary=read(root/'training_summary.json')
        alive=True
        try:os.kill(a.pid,0)
        except ProcessLookupError:alive=False
        if status['stop_reason'] and not summary and alive:
            (root/'STOP').touch()
        state='running' if alive else 'process exited; inspect log'
        if (root/'STOP').exists():state='stopping; saving checkpoint'
        if summary:state=summary.get('status') or ('stopped' if (root/'STOP').exists() else 'completed')
        state_step=(summary or runtime).get('timesteps',start)
        status.update(state=state,step=state_step,start=start,target=start+config.get('steps',0),workers=runtime.get('workers'),device=runtime.get('policy_device'),updated=time.time(),runtime_updated=runtime.get('updated_utc'),run=str(root))
        tmp=root/'guard.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(root/'guard.json')
        if summary or not alive:break
        time.sleep(15)
if __name__=='__main__':main()
