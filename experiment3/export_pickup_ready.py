"""Export a frozen throw contract from a separate reference simulation only."""
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
from pickup_common import ROOT, read_json, write_json, sha256, fresh_directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'experiment2/runs/20261101T152209002996/best_physics')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=fresh_directory(args.out);cfg=read_json(args.bundle/'config.json')
    hashes={}
    for name in ['policy.zip','normalize.pkl','config.json','evaluation.json']:
        hashes[name]=sha256(args.bundle/name);shutil.copy2(args.bundle/name,out/name)
    profile=cfg['throw_profile'];teacher=ROOT/profile['teacher_checkpoint']
    for name in ['policy.zip','normalize.pkl']:
        hashes['teacher/'+name]=sha256(teacher.parent/name)
        (out/'teacher').mkdir(exist_ok=True);shutil.copy2(teacher.parent/name,out/'teacher'/name)
    sys.path.insert(0,str(ROOT/'experiment2'))
    from env import build_env
    env=build_env(variant=cfg['variant'],throw_profile=profile,randomize=False,record=False)
    s,m,d=env.sim,env.m,env.d
    import mujoco
    r=d.xmat[s.hand].reshape(3,3)
    joint_targets={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.jnt_type[j] in [2,3]}
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in [ROOT/'inspire_model.py',ROOT/'inspire_grip.py',ROOT/'dart_model.py',ROOT/'dart_profile.json',ROOT/'experiment2/env.py',ROOT/'experiment2/calibrated_env.py',ROOT/'experiment2/parameter_env.py']}
    for name in sources:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write_json(out/'contract.json',dict(schema_version=1,source_bundle=str(args.bundle.relative_to(ROOT)),hashes=hashes,
                sources=sources,profile=profile,mujoco_version=mujoco.__version__,
                joint_targets=joint_targets,wrist_position=d.xpos[s.hand].tolist(),wrist_rotation=r.tolist(),
                dart_position_hand=(r.T@(d.xpos[s.dart]-d.xpos[s.hand])).tolist(),
                dart_rotation_hand=(r.T@d.xmat[s.dart].reshape(3,3)).tolist(),
                dart_mass=float(m.body_mass[s.dart]),dart_inertia=m.body_inertia[s.dart].tolist(),
                position_tolerance_m=.01,rotation_tolerance_deg=10,linear_speed_max=.02,angular_speed_max=.2,
                contact_requirement='thumb plus at least one index/middle finger',hold_seconds=2,
                note='Reference reset occurred ONLY in a separate throwing simulator. Never reset an executing pickup to this state.'))
    env.close();print('Frozen contract exported',flush=True)


if __name__=='__main__':main()
