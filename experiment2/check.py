"""Contract, reward gate, deterministic reset and real-physics episode checks."""
import json
from pathlib import Path
import numpy as np
import mujoco
from stable_baselines3.common.env_checker import check_env
from env import DartThrowEnv,impact_reward
from trajectory import guidance,release_score

def main():
    assert impact_reward(.0,True,True)==(140.,True)
    assert not impact_reward(.007,True,True)[1]
    assert impact_reward(0.,False,True)==(-15.,False)
    assert impact_reward(0.,True,False)==(-15.,False)
    # A weak floor-bound release still gets a graded teaching signal. Raising
    # vz toward the required arc must improve it; exact ballistic arcs peak.
    p=np.array([.5,.15,1.42]);goal=np.array([3.,0.,1.73]);v=np.array([3.2,0.,.94])
    low=guidance(p,v,goal);higher=guidance(p,v+[0,0,1.],goal)
    assert higher['velocity_quality']>low['velocity_quality']>0
    assert low['required_vz_at_current_vx_m_s']>4.
    exact=(goal-p)/.7+np.array([0,0,4.905*.7])
    best=guidance(p,exact,goal)
    assert best['velocity_gap_m_s']<1e-10 and best['predicted_board_error_m']<1e-10
    assert release_score(best,exact/np.linalg.norm(exact),exact)[0]>release_score(low,[1,0,0],v)[0]
    env=DartThrowEnv();check_env(env,warn=True)
    a,_=env.reset(seed=101);b,_=env.reset(seed=101);np.testing.assert_allclose(a,b,atol=1e-7)
    observations=[]
    # Collision fixtures only: directly initialize near the board to verify the
    # scoring gate. These are never training rollouts or reported learned hits.
    fixtures=[]
    for offset,released,expected in ((0.,True,True),(.03,True,False),(0.,False,False)):
        env.reset(seed=101);s=env.sim;d=env.d
        d.qpos[s.dart_q:s.dart_q+3]=[3.-s.dynamics.prop['tip_x_m']-.003,offset,1.73]
        d.qpos[s.dart_q+3:s.dart_q+7]=[1,0,0,0]
        d.qvel[s.dart_v:s.dart_v+6]=[3,0,0,0,0,0]
        env.released=released;env.commanded=released;s.closed=not released
        mujoco.mj_forward(env.m,d)
        _,reward,term,trunc,info=env.step(np.zeros(10,dtype=np.float32))
        assert term and info['outcome']=='board',info
        assert info['is_success']==expected,info
        fixtures.append({'offset_m':offset,'valid_release':released,'success':info['is_success'],'reward':reward})
    for seed in (101,102):
        env.reset(seed=seed);done=False
        while not done:
            action=np.zeros(10,dtype=np.float32);action[-1]=1 if env.t>.06 else -1
            obs,reward,term,trunc,info=env.step(action)
            assert np.isfinite(obs).all() and np.isfinite(reward);done=term or trunc
        observations.append(info)
    env.close()
    revised=DartThrowEnv(variant='arc_v2');check_env(revised,warn=True)
    for offset,released,expected in ((0.,True,True),(.03,True,False),(0.,False,False)):
        revised.reset(seed=101);s=revised.sim;d=revised.d
        d.qpos[s.dart_q:s.dart_q+3]=[3.-s.dynamics.prop['tip_x_m']-.003,offset,1.73]
        d.qpos[s.dart_q+3:s.dart_q+7]=[1,0,0,0]
        d.qvel[s.dart_v:s.dart_v+6]=[3,0,0,0,0,0]
        revised.released=released;revised.commanded=released;s.closed=not released
        mujoco.mj_forward(revised.m,d)
        _,_,term,_,info=revised.step(np.zeros(10,dtype=np.float32))
        assert term and info['outcome']=='board' and info['is_success']==expected,info
    revised.reset(seed=101)
    done=False
    while not done:
        action=np.zeros(10,dtype=np.float32);action[-1]=1 if revised.t>.06 else -1
        obs,reward,term,trunc,info=revised.step(action);done=term or trunc
        assert np.isfinite(obs).all() and np.isfinite(reward)
    assert info['variant']=='arc_v2' and info['outcome'] is not None
    if info['release'] is not None:
        assert info['trajectory']['velocity_gap_m_s']>=0
    # The untrained zero-action prior can lose its grasp: do not label that a
    # valid release just to pass a test. Checkpoint comparisons exercise release.
    revised.close()
    Path(__file__).with_name('checks.json').write_text(json.dumps({'passed':True,'synthetic_collision_fixtures':fixtures,'episodes':observations},indent=2))
    print('Environment/reward checks passed',flush=True)

if __name__=='__main__':main()
