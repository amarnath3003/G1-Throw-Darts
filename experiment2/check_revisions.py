"""Contract, motion derivative/limit, physics invariance and action sensitivity."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from stable_baselines3.common.env_checker import check_env
from env import DartThrowEnv
from physics_eval import physical_summary


def main():
    base=DartThrowEnv(variant='arc_v5');checks=[]
    urdf=ET.parse(Path(__file__).resolve().parents[1]/'assets/inspire/upstream_g1.urdf').getroot()
    for variant in ('arc_v6','arc_v7','arc_v8','arc_v9','arc_v10'):
        env=DartThrowEnv(variant=variant);check_env(env,warn=True)
        first,_=env.reset(seed=101);second,_=env.reset(seed=101)
        np.testing.assert_allclose(first,second,rtol=0,atol=1e-7)
        for field in ('body_mass','body_inertia','geom_friction','actuator_ctrlrange','jnt_range'):
            np.testing.assert_array_equal(getattr(env.m,field),getattr(base.m,field))
        low=env.residual_target(np.full(9,.4));high=env.residual_target(np.full(9,.8))
        assert np.all(high>low),variant
        assert np.all(np.abs(high)<=env.scale)
        ts=np.linspace(2.1,2.32,501);q=env.curve(ts)[:,env.qadr];v=env.curve(ts,1)[:,env.qadr]
        lim=env.m.jnt_range[env.jids];assert np.all(q>=lim[:,0]) and np.all(q<=lim[:,1])
        speeds=np.array([float(urdf.find(f"joint[@name='{n}']/limit").get('velocity')) for n in env.names])
        assert np.all(np.abs(v)<=speeds)
        t=2.177;h=1e-6
        np.testing.assert_allclose((env.curve(t+h)-env.curve(t-h))/(2*h),env.curve(t,1),atol=1e-5,rtol=1e-5)
        env.contact_event=lambda: (env.t<env.release_earliest-.005,[])
        while env.t<env.release_earliest:
            _,_,term,_,_=env.step(np.zeros(10,dtype=np.float32));assert not term
        action=np.zeros(10,dtype=np.float32);action[-1]=1
        _,_,term,_,info=env.step(action)
        assert term and info['outcome']=='premature_slip' and info['release'] is None
        checks.append({'variant':variant,'passed':True});env.close()
    base.close()
    clean={'release':{},'outcome':'environment_hit','closest_tip_distance_m':1.,'is_success':False}
    bad=dict(clean,outcome='robot_hit',closest_tip_distance_m=0.)
    assert tuple(physical_summary([clean]*8,2.6)['selection_score'])>tuple(physical_summary([bad]*7+[clean],2.6)['selection_score'])
    out=Path(__file__).resolve().parent/'analysis/round300k/continuation_20261030/revision_checks.json'
    out.write_text(json.dumps({'passed':True,'variants':checks,'physical_selection_guard':True},indent=2))
    print('Revised-control and physics-selection checks passed',flush=True)


if __name__=='__main__':main()
