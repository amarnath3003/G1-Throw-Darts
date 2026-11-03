"""Regression checks for reference calibration and release-credit handling."""
import json
from pathlib import Path
import numpy as np
from env import DartThrowEnv,build_env
from calibrated_env import CalibratedThrowEnv,measured_target_reward


def main():
    base=DartThrowEnv(variant='arc_v7',randomize=True)
    calibrated=build_env(variant='arc_v7',throw_profile={},randomize=True)
    np.testing.assert_array_equal(base.reset(seed=123)[0],calibrated.reset(seed=123)[0])
    times=np.linspace(2.1,2.32,1001)
    for order in (0,1):np.testing.assert_array_equal(base.curve(times,order),calibrated.curve(times,order))
    for field in ('body_mass','body_inertia','geom_friction','actuator_ctrlrange','jnt_range','geom_size'):
        np.testing.assert_array_equal(getattr(base.m,field),getattr(calibrated.m,field))
    profile=dict(elbow_back=-.1,shoulder_finish=.1,rate_scale=1.2,roll_finish=.1,elbow_finish=.2)
    calibrated.configure(profile)
    t=2.177;h=1e-6
    np.testing.assert_allclose((calibrated.curve(t+h)-calibrated.curve(t-h))/(2*h),calibrated.curve(t,1),atol=1e-5,rtol=1e-5)
    info=dict(release={},outcome='environment_hit',reward_terms={},board_plane_crossing=dict(error_m=.1))
    assert measured_target_reward(info)>0
    bad=dict(info,outcome='recontact');assert measured_target_reward(bad)==0
    bad=dict(info,outcome='board',impact_error_m=0.,reward_terms=dict(board_impact=-15.))
    assert measured_target_reward(bad)==0
    values=[measured_target_reward(dict(info,board_plane_crossing=dict(error_m=e))) for e in (0.,.00635,.08,.4,1.)]
    assert all(a>b for a,b in zip(values,values[1:]))
    # Test accumulation/contact termination using deterministic transitions,
    # not a fake performance throw or an unrecorded physical swing.
    original=DartThrowEnv.step;calls=[]
    def fixture(self,action):
        self.t+=.02;self.released=True;calls.append(np.asarray(action).copy())
        done=len(calls)==3
        row=dict(info,outcome='environment_hit' if done else None)
        row['reward_terms']={};return np.zeros(73),1.,done,False,row
    DartThrowEnv.step=fixture
    try:
        calibrated.profile.update(flight_autopilot=True,target_reward_weight=1.)
        calibrated.t=0.;calibrated.terms={};calibrated.episode_return=0.
        _,reward,done,_,result=calibrated.step(np.zeros(10))
        assert done and len(calls)==3 and calls[1][-1]==1
        np.testing.assert_allclose(reward,3.+measured_target_reward(info))
        np.testing.assert_allclose(sum(x[2] for x in calibrated.transition_controls),reward)
        assert result['outcome']=='environment_hit' and not result.get('is_success',False)
    finally:DartThrowEnv.step=original;base.close();calibrated.close()
    root=Path(__file__).parent/'analysis/round400k';root.mkdir(exist_ok=True,parents=True)
    (root/'calibration_checks.json').write_text(json.dumps(dict(passed=True,
        checks=['zero-profile equivalence','unchanged physical parameters','reference derivatives',
                'invalid-contact reward exclusion','monotonic actual aim reward','terminal flight credit and control archive']),indent=2))
    print('Calibration checks passed')


if __name__=='__main__':main()
