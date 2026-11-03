"""Remote physical equivalence, reward/guard checks and sensitivity probes."""
import runtime_bootstrap
import argparse,json
from pathlib import Path
import numpy as np
import torch
from train import Archive
from env import build_env
from parameter_env import precision_objective, PARAMETERS
from throw_diagnostics import diagnostics
from remote_training.guard import assess

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=a.reuse)
    profile=json.loads((Path(__file__).parent/'remote_training/parameter_profile.json').read_text())
    env=build_env(variant='arc_v7',throw_profile=profile,randomize=True,record=True)
    archive=Archive(env,a.output/'episodes');rows=[]
    try:
        # Baseline seeds are only equivalence checks. Sensitivity uses separate
        # diagnostic seeds; neither is an independent final policy test set.
        tests=[('zero',seed,np.zeros(8)) for seed in range(83101,83133)]
        for i,name in enumerate(PARAMETERS):
            for sign in (-1,1):
                action=np.zeros(8);action[i]=sign*.5
                for seed in (85101,85102):tests.append((name+str(sign),seed,action))
        for index,(label,seed,action) in enumerate(tests):
            if a.reuse:
                folder=a.output/'episodes'/f'{index:06d}'
                info=json.loads((folder/'result.json').read_text())
                reward=info['parameter_reward'];term=True;trunc=False
                control_sum=np.load(folder/'actions.npz')['reward'].sum()
            else:
                archive.reset(seed=seed)
                _,reward,term,trunc,info=archive.step(action)
                control_sum=sum(r[2] for r in env.transition_controls)
            assert term and not trunc
            assert abs(sum(info['reward_terms'].values())-reward)<1e-6
            assert abs(control_sum-reward)<1e-6
            assert info['max_joint_limit_violation_rad']<.03
            row=dict(info,seed=seed,label=label);rows.append(row)
            print(json.dumps(dict(label=label,seed=seed,outcome=info['outcome'],error=info['impact_error_m'],reward=reward)),flush=True)
    finally:archive.close()
    zeros=rows[:32]
    source=Path(profile['teacher_checkpoint']).parent/'evaluation.json'
    old=json.loads(source.read_text())['episodes']
    assert [r['outcome'] for r in zeros]==[r['outcome'] for r in old]
    delta=max(abs(r['closest_tip_distance_m']-s['closest_tip_distance_m']) for r,s in zip(zeros,old))
    assert delta<1e-6,delta # CPU teacher versus GPU reference: sub-micrometre tolerance
    # The new objective must favor accuracy, and reject all body collisions.
    hit=next(r for r in zeros if r['reward_terms'].get('board_impact',0)>0)
    vals=[]
    for error in (.20,.10,.05,.02,.006):
        h=dict(hit,impact_error_m=error,is_success=error<=.00635)
        vals.append(precision_objective(h,np.zeros(8)))
    assert np.all(np.diff(vals)>0)
    assert precision_objective(dict(hit,outcome='robot_hit'),np.zeros(8))<0
    # Small transient dips do not terminate, sustained deterioration does.
    def report(step,d,clean=1.):
        return dict(timesteps=step,summary=dict(episodes=32,bullseyes=0,valid_board_hits=29,clean_flight_rate=clean,
                    mean_penalized_closest_distance_m=d,mean_capped_target_error_m=d,selection_score=[1,-d,0,29]))
    assert assess([report(0,.15),report(768,.22),report(1536,.21)])['stop_reason'] is None
    assert assess([report(0,.15)]+[report(768*i,.23) for i in range(1,7)])['stop_reason']
    assert assess([report(0,.15)]+[report(768*i,d) for i,d in enumerate([.27,.26,.25,.24,.22,.20],1)])['stop_reason'] is None
    assert assess([report(0,.15),report(768,.3,.5),report(1536,.3,.5)])['stop_reason']
    result=dict(zero_equivalence_max_error_m=delta,baseline=diagnostics(zeros),
        sensitivity={label:diagnostics([r for r in rows if r['label']==label]) for label in sorted({r['label'] for r in rows})},episodes=rows)
    (a.output/'report.json').write_text(json.dumps(result,indent=2))
    print('CHECKS PASSED '+json.dumps(result['baseline']),flush=True)

if __name__=='__main__':main()
