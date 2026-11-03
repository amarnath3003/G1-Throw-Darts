"""Independent hold/release checks for the learned left-hand controller."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import mujoco
import numpy as np

from inspire_grip import InspireGrip, OUT
from train_inspire_grip import episode


def random_case(args):
    checkpoint,seed=args
    s=InspireGrip(checkpoint)
    return episode(s,seed,duration=6.)


def trace_case(checkpoint, duration=60., timestep=.0005, disabled=False, initially_open=False):
    s=InspireGrip(checkpoint,timestep=timestep)
    m,d=s.model,s.data
    if disabled:s.hand_enabled=False
    if initially_open:
        s.place(s.config['q_open'],s.config['dart_pos_hand'])
        s.command=d.qpos[s.fq].copy();s.closed=False
        mujoco.mj_forward(m,d)
        s.initial_relative=s.relative_pos().copy()
    all_hand_j=[j for j in range(m.njnt) if (m.joint(j).name or '').startswith('L_')]
    all_q=m.jnt_qposadr[all_hand_j];all_v=m.jnt_dofadr[all_hand_j]
    eq1=m.jnt_qposadr[m.eq_obj1id];eq2=m.jnt_qposadr[m.eq_obj2id]
    rows=[];max_torque=0.;max_velocity=0.;limit_violation=0.;coupling_error=0.
    initial=s.sample();initial_z=initial['com'][2]
    for i in range(round(duration/timestep)):
        s.step()
        max_torque=max(max_torque,float(np.max(np.abs(d.ctrl[s.fa]))))
        max_velocity=max(max_velocity,float(np.max(np.abs(d.qvel[all_v]))))
        limit_violation=max(limit_violation,float(np.max(np.maximum(
            m.jnt_range[all_hand_j,0]-d.qpos[all_q],d.qpos[all_q]-m.jnt_range[all_hand_j,1]))))
        coupling_error=max(coupling_error,float(np.max(np.abs(d.qpos[eq1]-m.eq_data[:,1]*d.qpos[eq2]-m.eq_data[:,0]))))
        if i%max(1,round(.02/timestep))==0:rows.append(s.sample())
    held_rows=rows.copy();release=s.sample();s.closed=False
    release_peak_speed=0.;release_self_penetration=0.
    for i in range(round(2./timestep)):
        s.step()
        release_peak_speed=max(release_peak_speed,float(np.max(np.abs(d.qvel[all_v]))))
        if i%20==0:
            for c in d.contact:
                b1,b2=m.geom_bodyid[[c.geom1,c.geom2]]
                if m.body(b1).name.startswith('L_') and m.body(b2).name.startswith('L_'):
                    release_self_penetration=max(release_self_penetration,-float(c.dist))
        if i%max(1,round(.005/timestep))==0:rows.append(s.sample())
    after=[r for r in rows if r['time_s']>duration+.005]
    cleared=next((r['time_s']-duration for r in after if max(r['normal_N'])<.01
                  and not r['other_contacts'] and r['com'][2]<release['com'][2]-.001),None)
    at_half=min(after,key=lambda r:abs(r['time_s']-duration-.5))
    steady=[r for r in held_rows if r['time_s']>1.]
    result=dict(duration_s=duration,timestep_s=timestep,
                max_hold_drift_m=max(r['relative_drift_m'] for r in held_rows),
                hold_height_drop_m=initial_z-release['com'][2],
                minimum_steady_normal_N=min((min(r['normal_N']) for r in steady),default=0.),
                unwanted_hold_contacts=sorted(set(n for r in held_rows for n in r['other_contacts'])),
                maximum_penetration_m=max(r['penetration_m'] for r in held_rows),
                peak_motor_torque_Nm=max_torque,peak_actual_joint_speed_rad_s=max_velocity,
                maximum_joint_limit_violation_rad=limit_violation,
                maximum_coupling_residual_rad=coupling_error,
                release_clear_s=cleared,release_drop_at_half_second_m=release['com'][2]-at_half['com'][2],
                release_peak_joint_speed_rad_s=release_peak_speed,release_self_penetration_m=release_self_penetration,
                warnings=d.warning.number.tolist(),finite=bool(np.isfinite(d.qpos).all()),
                final_hold=release)
    result['passed']=(result['max_hold_drift_m']<.008 and result['minimum_steady_normal_N']>.03
                      and not result['unwanted_hold_contacts'] and result['maximum_penetration_m']<.0005
                      and cleared is not None and cleared<.3 and result['release_drop_at_half_second_m']>.2
                      and max_torque<=.350001 and release_peak_speed<.65 and release_self_penetration<.0001
                      and limit_violation<.001 and coupling_error<.001 and result['finite'] and not any(result['warnings']))
    return result,rows


def validate(checkpoint=OUT/'policy.json'):
    checkpoint=str(checkpoint)
    digest=hashlib.sha256(open(checkpoint,'rb').read()).hexdigest()
    long,rows=trace_case(checkpoint)
    print('60-second hold',long['passed'],'drift_mm',long['max_hold_drift_m']*1000,flush=True)
    (OUT/'trace.json').write_text(json.dumps(rows,indent=2))
    fine,_=trace_case(checkpoint,duration=8.,timestep=.00025)
    print('half timestep',fine['passed'],flush=True)
    unpowered,_=trace_case(checkpoint,duration=.6,disabled=True)
    opened,_=trace_case(checkpoint,duration=.4,initially_open=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        random_rows=list(pool.map(random_case,[(checkpoint,i) for i in range(101,125)]))
    random_pass=sum(r['held'] and r['released'] for r in random_rows)
    negative_pass=unpowered['hold_height_drop_m']>.10 and opened['hold_height_drop_m']>.10
    report=dict(hand='left',embodiment='G1 with five-finger Inspire DFQ',policy_sha256=digest,
                long_hold=long,half_timestep=fine,unpowered=unpowered,initially_open=opened,
                randomization=dict(seeds=list(range(101,125)),mass_scale=[.9,1.1],friction=[.55,.9],
                                   placement_each_world_axis_m=[-.00015,.00015],pulse_force_N=.06,
                                   pulse_duration_s=.15,duration_s=6.,successes=random_pass,total=len(random_rows)),
                random_cases=random_rows,negative_controls_passed=negative_pass,
                passed=bool(long['passed'] and fine['passed'] and negative_pass and random_pass==len(random_rows)))
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['randomization','negative_controls_passed','passed']},indent=2),flush=True)
    if not report['passed']:raise SystemExit('Validation failed; inspect validation.json')


if __name__=='__main__':
    validate()
