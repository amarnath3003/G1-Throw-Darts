"""Fit, train, validate and display the photo-guided five-finger grasp."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from inspire_grip import InspireGrip, OUT, NAMES, DISTAL


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def fit_pose():
    s=InspireGrip(OUT/'left_seed.json'); m,d=s.model,s.data
    bars=[m.geom(f"spare_dart_1_barrel_{i}").id for i in range(24)]
    gids=[m.geom(n+"_collision_0").id for n in DISTAL]
    others=[i for i in range(m.ngeom) if (m.body(m.geom_bodyid[i]).name or '').startswith('L_')
            and m.geom_contype[i] and i not in gids]
    nonbar=[i for i in range(m.ngeom) if m.geom_bodyid[i]==s.dart and i not in bars]
    # Anatomical topology from the user's photos, fitted to the real link surfaces.
    # The barrel crosses the index/middle fingers; the thumb approaches the other side.
    quat=[2**-.5,0,-2**-.5,0]
    def apply(x):
        s.place([*x[:4],1.35,1.35], x[4:7],quat)
    def distance(g):
        distances=[mujoco.mj_geomDistance(m,d,g,b,.2,None) for b in bars]
        b=bars[int(np.argmin(distances))]; line=np.zeros(6)
        dist=mujoco.mj_geomDistance(m,d,g,b,.2,line)
        return dist,line
    def loss(x):
        apply(x)
        nearest=[distance(g) for g in gids]
        residual=[(a[0]-.00004)*1000 for a in nearest]
        points=[d.xmat[s.hand].reshape(3,3).T@(a[1][:3]-d.xpos[s.hand]) for a in nearest]
        residual.append((points[0][2]-(points[1][2]+points[2][2])/2)*150)
        # Keep the grasp on the pads toward the ends of the actual finger meshes.
        tip_centers=[np.array([.015,-.001,-.006]),np.array([.001,-.032,-.005]),np.array([.001,-.035,-.005])]
        for body,(dist,line),local in zip(s.distal,nearest,tip_centers):
            point_local=d.xmat[body].reshape(3,3).T@(line[:3]-d.xpos[body])
            residual.extend((point_local-local)*40)
        for g in others:
            residual.append(min(distance(g)[0]-.0001,0)*1500)
        for g in gids+others:
            residual.append(min(min(mujoco.mj_geomDistance(m,d,g,b,.2,None) for b in nonbar)-.0001,0)*1500)
        # Regularization chooses one feasible pose without forcing human proportions.
        residual.extend((x-seed) * [0.03,.03,.03,.03,1.,1.,1.])
        return residual
    lower=[0.,.01,.1,.1,.145,-.10,-.005]
    upper=[1.3,.495,1.35,1.35,.225,-.025,.045]
    seed=np.r_[s.config['q_grasp'][:4],s.config['dart_pos_hand']]
    best=None
    rng=np.random.default_rng(20261028)
    history=[]
    for k in range(8):
        x=seed.copy() if k==0 else np.clip(seed+rng.normal(size=7)*[.08,.03,.04,.04,.002,.002,.004],np.array(lower)+1e-6,np.array(upper)-1e-6)
        result=least_squares(loss,x,bounds=(lower,upper),max_nfev=180,diff_step=1e-4,
                             ftol=1e-6,xtol=1e-7,gtol=1e-6)
        value=float(np.linalg.norm(loss(result.x)))
        history.append(dict(start=k,loss=value,parameters=result.x.tolist()))
        print('fit',k,'loss',round(value,4),'x',result.x.round(4),flush=True)
        if best is None or value<best[0]:best=(value,result.x.copy())
    apply(best[1]);s.config.update(q_grasp=d.qpos[s.fq].tolist(),dart_pos_hand=best[1][4:7].tolist(),dart_quat_hand=quat,
                                  closure_direction=[0,-1,1,1,0,0])
    save(OUT/'pose_fit.json',dict(history=history,config=s.config))
    save(OUT/'left_pose.json',s.config)
    snapshot(s,OUT/'fit.png')
    return s


def snapshot(s,path,azimuth=260,elevation=-25,distance=.42):
    from PIL import Image
    m,d=s.model,s.data
    r=mujoco.Renderer(m,720,1000);c=mujoco.MjvCamera()
    c.lookat[:]=d.xpos[s.hand]+d.xmat[s.hand].reshape(3,3)@np.asarray(s.config['dart_pos_hand'])
    c.distance=distance;c.azimuth=azimuth;c.elevation=elevation
    option=mujoco.MjvOption();option.geomgroup[3]=0
    r.update_scene(d,c,scene_option=option)
    Image.fromarray(r.render()).save(path);r.close()


def train():
    fit_pose()


def validate(checkpoint):
    from validate_inspire_grip import validate as run
    run(checkpoint)


def render(checkpoint):
    from inspire_grip_view import record
    record(checkpoint)


def viewer(checkpoint):
    from inspire_grip_view import viewer as run
    run(checkpoint)
