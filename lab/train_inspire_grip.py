"""Seeded cross-entropy policy search for the left Inspire grip force controller.

This trains seven controller parameters in MuJoCo. It is not PPO or image-to-action
learning. The photo-guided contact topology and coupled-joint IK supply the seed.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from inspire_grip import InspireGrip, OUT

LOW = np.array([1.4,.6,.6,.01,.008,3.,.06])
HIGH = np.array([2.8,1.4,1.4,.10,.035,7.,.16])
LABELS = ['thumb_force_N','index_force_N','middle_force_N','force_gain_rad_Ns',
          'initial_squeeze_rad','kp_Nm_rad','kd_Nm_s_rad']
SIM = None
BASE = None


def init_worker(base):
    global SIM, BASE
    BASE=base
    SIM=InspireGrip(OUT/'left_candidate.json')


def configure(base, values):
    config=deepcopy(base)
    config.update(force_feedback=True,force_targets=list(values[:3]),force_gain=float(values[3]),
                  squeeze=float(values[4]),kp=float(values[5]),kd=float(values[6]))
    return config


def episode(s, seed, duration=5., nominal=False):
    m,d=s.model,s.data
    rng=np.random.default_rng(seed)
    mass_scale=1. if nominal else rng.uniform(.9,1.1)
    friction=.7 if nominal else rng.uniform(.55,.9)
    m.body_mass[s.dart]=.0267*mass_scale
    # Consistent mass scaling; geometry and COM stay unchanged.
    from dart_model import properties,load_profile
    m.body_inertia[s.dart]=np.array(properties(load_profile())['inertia_kg_m2'])*mass_scale
    for g in range(m.ngeom):
        if (m.body(m.geom_bodyid[g]).name or '').startswith('L_') and m.geom_contype[g]:
            m.geom_friction[g,0]=friction
    mujoco.mj_setConst(m,d)
    perturb=np.zeros(3) if nominal else rng.uniform(-.00015,.00015,3)
    s.reset(perturb=perturb)
    initial=s.sample()
    max_drift=0.;bad=0;min_normal=100.;max_force=0.;max_pen=initial['penetration_m']
    tail_start=None
    force=rng.normal(size=3);force=force/(np.linalg.norm(force)+1e-9)*.06
    steps=round(duration/m.opt.timestep)
    for i in range(steps):
        # A documented test disturbance, not a support or attachment force.
        d.qfrc_applied[s.dart_v:s.dart_v+3]=force if not nominal and 1.5<=d.time<1.65 else 0
        s.step()
        if i%40==0:
            row=s.sample();max_drift=max(max_drift,row['relative_drift_m'])
            max_force=max(max_force,max(row['normal_N']));max_pen=max(max_pen,row['penetration_m'])
            bad+=bool(row['other_contacts'])
            if d.time>1.:min_normal=min(min_normal,min(row['normal_N']))
            if d.time>=duration-1. and tail_start is None:tail_start=s.relative_pos().copy()
            if max_drift>.035 or not np.isfinite(d.qpos).all():
                d.qfrc_applied[:]=0
                return dict(cost=200.+min(max_drift,1.)*100,held=False,released=False,
                            drift_m=max_drift,seed=seed,friction=friction,mass_scale=mass_scale)
    d.qfrc_applied[:]=0
    end=s.sample();tail_drift=np.linalg.norm(s.relative_pos()-tail_start)
    s.closed=False;release_z=end['com'][2];clear_time=None
    for i in range(round(.65/m.opt.timestep)):
        s.step()
        if i%20==0:
            row=s.sample()
            if max(row['normal_N'])<.01 and not row['other_contacts'] and clear_time is None:
                clear_time=(i+1)*m.opt.timestep
    row=s.sample();drop=release_z-row['com'][2]
    released=clear_time is not None and clear_time<.3 and drop>.25
    held=max_drift<.008 and bad==0 and min_normal>.03
    cost=(max_drift*1000+tail_drift*5000+bad*.3+max(0,.15-min_normal)*30+
          max(0,max_force-4.)*10+max(0,max_pen-.0005)*10000+(0 if released else 100))
    return dict(cost=float(cost),held=bool(held),released=bool(released),drift_m=float(max_drift),
                tail_drift_m=float(tail_drift),min_normal_N=float(min_normal),peak_normal_N=float(max_force),
                unwanted_contact_samples=bad,release_clear_s=clear_time,drop_m=float(drop),
                seed=seed,friction=float(friction),mass_scale=float(mass_scale),penetration_m=float(max_pen))


def evaluate(values):
    SIM.config=configure(BASE,values)
    rows=[episode(SIM,seed) for seed in (11,29)]
    cost=float(np.mean([r['cost'] for r in rows])+np.max([r['cost'] for r in rows])*.3)
    return cost,rows


def train():
    base=json.loads((OUT/'left_candidate.json').read_text())
    rng=np.random.default_rng(20261028)
    mean=np.array([2.,1.,1.,.04,.025,4.,.10]);std=(HIGH-LOW)*.22
    history=[];best=None;started=time.time()
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(base,)) as pool:
        for generation in range(6):
            candidates=np.clip(rng.normal(mean,std,(12,7)),LOW,HIGH)
            candidates[0]=mean
            if best is not None:candidates[1]=best[1]
            results=list(pool.map(evaluate,candidates))
            ranking=np.argsort([r[0] for r in results])
            for i,(cost,rows) in enumerate(results):
                history.append(dict(generation=generation,parameters=candidates[i].tolist(),cost=cost,episodes=rows))
                if best is None or cost<best[0]:best=(cost,candidates[i].copy(),rows)
            elite=candidates[ranking[:4]]
            mean=.25*mean+.75*elite.mean(0)
            std=np.maximum(elite.std(0)*.9,(HIGH-LOW)*.035)
            report=dict(method='cross_entropy_policy_search',hand='left',seed=20261028,base_configuration=base,
                        labels=LABELS,training_seeds=[11,29],elapsed_s=time.time()-started,history=history)
            (OUT/'training.json').write_text(json.dumps(report,indent=2))
            policy=configure(base,best[1]);policy.update(training_method='cross_entropy_policy_search',
                                                        training_cost=best[0],hand='left')
            final_pose=OUT/'final_pose.json'
            if final_pose.exists():
                policy.update(json.loads(final_pose.read_text()))
            (OUT/'policy.json').write_text(json.dumps(policy,indent=2))
            print('generation',generation,'best_cost',round(best[0],4),'parameters',best[1].round(4),
                  'holds',[r['held'] for r in best[2]],'releases',[r['released'] for r in best[2]],flush=True)
    print('finished',round(time.time()-started,1),'seconds',flush=True)


if __name__=='__main__':
    train()
