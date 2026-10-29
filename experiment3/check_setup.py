"""Reproducible reset, reachability and flat-table clearance audit (no pickup)."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image

from env import PickupScene, ARM, geom_z_bounds


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()


def audit(timestep, render=False):
    s=PickupScene(timestep)
    m,d=s.model,s.data
    reset_q=d.qpos.copy()
    contacts=s.contacts()
    baseline_pairs={tuple(sorted((c['geom1'],c['geom2']))) for c in contacts if c['distance_m']<0}
    dart_contacts=[c for c in contacts if 'spare_dart_1' in c['geom1'] or 'spare_dart_1' in c['geom2']]
    empty=all('table_top' in (c['geom1'],c['geom2']) for c in dart_contacts)
    speed=float(np.linalg.norm(d.qvel[s.dv:s.dv+3]))
    angular=float(np.linalg.norm(d.qvel[s.dv+3:s.dv+6]))
    lower=min(geom_z_bounds(m,d,g)[0] for g in s.dart_geoms)
    result=dict(timestep_s=m.opt.timestep, reset=dict(empty_hand=empty,
                dart_table_contact=any('table_top' in (c['geom1'],c['geom2']) for c in dart_contacts),
                dart_speed_m_s=speed, dart_angular_speed_rad_s=angular,
                lowest_dart_clearance_m=lower-s.config['table_top_z'],contacts=dart_contacts,
                dart_position=d.xpos[s.dart].tolist(), dart_quat_wxyz=d.xquat[s.dart].tolist()))
    # Reach target is the same hand-relative barrel transform as the proven hold.
    desired_rot=d.xmat[s.dart].reshape(3,3) @ rotation(s.grasp['dart_quat_hand']).T
    desired_pos=d.xpos[s.dart]-desired_rot@np.asarray(s.grasp['dart_pos_hand'])
    seed=d.qpos[s.aq].copy()
    lo,hi=m.jnt_range[s.arm_ids].T
    hand_geoms=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('L_') and m.geom_contype[g]]
    table=m.geom('table_top').id

    def solve(pos):
        def residual(q):
            d.qpos[s.aq]=q
            mujoco.mj_forward(m,d)
            r=d.xmat[s.hand].reshape(3,3)
            return np.r_[10*(d.xpos[s.hand]-pos), Rotation.from_matrix(desired_rot.T@r).as_rotvec()]
        rng=np.random.default_rng(731)
        candidates=[]
        for initial in [seed]+[rng.uniform(lo+.001,hi-.001) for _ in range(7)]:
            fit=least_squares(residual,np.clip(initial,lo+1e-6,hi-1e-6),bounds=(lo+1e-7,hi-1e-7),max_nfev=160)
            candidates.append(fit)
        fit=min(candidates,key=lambda f: np.linalg.norm(f.fun))
        residual(fit.x)
        return dict(joint_positions=dict(zip(ARM,fit.x.tolist())),
                    position_error_m=float(np.linalg.norm(d.xpos[s.hand]-pos)),
                    rotation_error_deg=float(np.rad2deg(np.linalg.norm(fit.fun[3:]))))

    if render:
        render_image(s,'reset.png')
    for name,height,fingers in [('pregrasp',.12,s.grasp['q_open']),('terminal_grip_on_table',0.,s.grasp['q_grasp'])]:
        d.qpos[:]=reset_q
        d.qpos[s.fq]=fingers
        s.couple()
        offset=np.array([0,.08,height]) if name=='pregrasp' else np.zeros(3)
        item=solve(desired_pos+offset)
        item['approach_offset_world_m']=offset.tolist()
        distance=[]
        for g in hand_geoms:
            dist=mujoco.mj_geomDistance(m,d,g,table,1.,None)
            distance.append((float(dist),m.geom(g).name))
        item['minimum_hand_table_distance_m'],item['closest_hand_geom']=min(distance)
        item['fingertip_clearance_m']={body:min(geom_z_bounds(m,d,g)[0] for g in hand_geoms if m.geom_bodyid[g]==m.body(body).id)-s.config['table_top_z']
                                     for body in ['L_thumb_distal','L_index_intermediate','L_middle_intermediate']}
        item['penetrating_contacts']=[c for c in s.contacts() if c['distance_m'] < -.0002 and 'spare_dart_1' not in c['geom1'] and 'spare_dart_1' not in c['geom2']]
        item['new_penetrating_contacts']=[c for c in item['penetrating_contacts'] if tuple(sorted((c['geom1'],c['geom2']))) not in baseline_pairs]
        item['kinematically_reachable']=item['position_error_m']<.005 and item['rotation_error_deg']<5
        item['table_clear']=item['minimum_hand_table_distance_m']>=-.0002
        result[name]=item
        if render:
            render_image(s,name+'.png')
    s.reset()
    result['reset_repeat_max_qpos_difference']=float(np.max(np.abs(d.qpos-reset_q)))
    result['reset_pass']=bool(empty and result['reset']['dart_table_contact'] and speed<.001 and angular<.01 and lower-s.config['table_top_z']>-.001 and result['reset_repeat_max_qpos_difference']<1e-10)
    return result


def render_image(s,name):
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[.43,.34,.96]
    camera.distance=1.55
    camera.azimuth=135
    camera.elevation=-22
    with mujoco.Renderer(s.model,height=720,width=960) as renderer:
        renderer.update_scene(s.data,camera=camera)
        Image.fromarray(renderer.render()).save(Path(__file__).resolve().parent/'research'/name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    results=[audit(.0005,args.render),audit(.00025)]
    path=Path(__file__).resolve().parent/'research/setup_audit.json'
    path.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(json.dumps({k:v for k,v in r.items() if k not in ('reset',)},indent=2))
    if not all(r['reset_pass'] and r['pregrasp']['kinematically_reachable'] and r['pregrasp']['table_clear'] and not r['pregrasp']['new_penetrating_contacts'] for r in results):
        raise SystemExit('Setup validation failed; inspect research/setup_audit.json')
