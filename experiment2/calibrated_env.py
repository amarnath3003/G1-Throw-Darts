"""Versioned, auditable joint-reference calibration around the arc_v7 policy.

Only robot joint references/rate limits change. Contacts, dart dynamics and
success checks are inherited unchanged. This is an optimizer-assisted PPO
warm start, not a different projectile simulator.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from env import DartThrowEnv


DEFAULT_PROFILE = dict(elbow_back=0., shoulder_finish=0., rate_scale=1.,
                       roll_finish=0., elbow_finish=0., wrist_finish=0., ramp_seconds=.08,
                       aim_yaw=0.,flight_autopilot=False, target_reward_weight=0.,
                       proxy_reward_weight=1., invalid_return_cap=0.)


def terminal_corrections(info, profile, episode_return):
    """Versioned reward changes only; preserve raw terms for audit."""
    proxy=sum(info['reward_terms'].get(k,0.) for k in ('release_arc','arc_improvement'))
    correction=(profile['proxy_reward_weight']-1.)*proxy
    valid=info.get('release') is not None and info['outcome'] in ('board','environment_hit','timeout')
    if info['outcome']=='board' and info['reward_terms'].get('board_impact',0)<=0:valid=False
    cap=profile['invalid_return_cap']
    penalty=min(0.,cap-(episode_return+correction)) if cap<0 and not valid else 0.
    return correction,penalty


def measured_target_reward(info):
    """Broad-to-fine actual aim reward, never a synthetic hit declaration."""
    if info.get('release') is None:return 0.
    if info['outcome']=='board':
        if info['reward_terms'].get('board_impact',0)<=0:return 0.
        error=info['impact_error_m']
    elif info['outcome'] in ('environment_hit','timeout') and info.get('board_plane_crossing'):
        error=info['board_plane_crossing']['error_m']
    else:return 0.
    return float(sum(weight/(1+(error/width)**2) for weight,width in ((16.,.4),(8.,.08),(4.,.02))))


class CalibratedThrowEnv(DartThrowEnv):
    def __init__(self, profile=None, **kwargs):
        self.profile = DEFAULT_PROFILE.copy()
        kwargs.pop('variant', None)
        super().__init__(variant='arc_v7', **kwargs)
        self.base_plan = deepcopy(self.plan)
        self.base_curve = self.curve
        self.base_slew = np.asarray(self.slew).copy()
        urdf = ET.parse(Path(__file__).resolve().parents[1]/'assets/inspire/upstream_g1.urdf').getroot()
        self.reference_speed_limits = np.array([float(urdf.find(f"joint[@name='{n}']/limit").get('velocity')) for n in self.names])
        self.configure(profile or {})

    def configure(self, profile):
        if set(profile) - set(DEFAULT_PROFILE):
            raise ValueError('Unknown throw-profile field')
        self.profile = dict(DEFAULT_PROFILE, **profile)
        if not all(np.isfinite(v) for v in self.profile.values()):
            raise ValueError('Nonfinite throw profile')
        if not .03 <= self.profile['ramp_seconds'] <= .2 or not .5 <= self.profile['rate_scale'] <= 2.:
            raise ValueError('Invalid reference-rate calibration')
        if not isinstance(self.profile['flight_autopilot'], bool) or not 0<=self.profile['target_reward_weight']<=4:
            raise ValueError('Invalid training profile')
        if abs(self.profile['aim_yaw'])>.35:raise ValueError('Aim yaw exceeds calibration range')
        if not 0<=self.profile['proxy_reward_weight']<=1 or not -100<=self.profile['invalid_return_cap']<=0:
            raise ValueError('Invalid reward correction profile')
        self.plan = deepcopy(self.base_plan)
        times = np.asarray(self.plan['times'])
        rows = self.base_curve(times); slopes = self.base_curve(times, 1)
        start, end = self.plan['start'], self.plan['end']
        u = np.clip((times-start)/(end-start), 0, 1)
        h = u*u*(3-2*u); dh = 6*u*(1-u)/(end-start)
        for index, back, finish in (
            (0, 0., self.profile['shoulder_finish']),
            (1, 0., self.profile['roll_finish']),
            (3, self.profile['elbow_back'], self.profile['elbow_finish']),
            (5, 0., self.profile['wrist_finish'])):
            address = self.qadr[index]
            rows[:, address] += back*(1-h)+finish*h
            slopes[:, address] += (finish-back)*dh
            self.plan['arm_keyframes'][1][index] += back
            self.plan['arm_keyframes'][2][index] += finish
        rows[:,self.qadr[7]] += self.profile['aim_yaw']
        self.curve = CubicHermiteSpline(times, rows, slopes)
        self.slew = self.base_slew.copy()
        self.slew[[0,3,5]] *= self.profile['rate_scale']
        for name, value in zip(self.names[:7], self.plan['arm_keyframes'][1]):
            self.sim.config['arm_pose'][name] = value
        self.sim.config['arm_pose']['waist_yaw_joint'] = -.025+self.profile['aim_yaw']
        # Validate complete reference, including inter-keyframe overshoot.
        samples = np.linspace(start, end, 1001)
        q = self.curve(samples)[:,self.qadr]
        limits = self.m.jnt_range[self.jids]
        if np.any(q < limits[:,0]+.04) or np.any(q > limits[:,1]-.04):
            raise ValueError('Calibrated reference violates joint margins')
        # Conservative bound includes the full possible residual-rate command.
        speeds = np.abs(self.curve(samples,1)[:,self.qadr])+np.maximum(.06,self.slew)/.02
        if np.any(speeds > self.reference_speed_limits):
            raise ValueError('Calibrated reference exceeds joint speed limit')

    def slew_limit(self):
        blend = float(np.clip((self.t-.02)/self.profile['ramp_seconds'], 0, 1))
        return .06+blend*(self.slew-.06)

    def step(self, action):
        self.transition_controls=[]
        def advance(control):
            start=self.t
            result=super(CalibratedThrowEnv,self).step(control)
            self.transition_controls.append((start,np.asarray(control).copy(),float(result[1]),
                self.sim.target.copy(),self.sim.velocity_target.copy(),self.d.ctrl.copy()))
            return result
        obs,reward,term,trunc,info = advance(action)
        # Once detached, give this release decision its actual terminal result.
        # The simulator still steps every 0.5 ms, checks every collision and
        # records every 20 ms. Robot follow-through uses the unchanged prior
        # with held residual offsets; no dart state is set or extrapolated.
        if self.profile['flight_autopilot'] and self.released:
            follow=np.zeros(10);follow[-1]=1.
            while not (term or trunc):
                obs,r,term,trunc,info=advance(follow);reward+=r
        if term or trunc:
            extra=self.profile['target_reward_weight']*measured_target_reward(info)
            reward+=self.add('measured_target',extra);self.episode_return+=extra
            correction,penalty=terminal_corrections(info,self.profile,self.episode_return)
            reward+=self.add('proxy_correction',correction)+self.add('invalid_return_correction',penalty)
            self.episode_return+=correction+penalty
            extra+=correction+penalty
            row=self.transition_controls[-1]
            self.transition_controls[-1]=(*row[:2],row[2]+extra,*row[3:])
            info['reward_terms']=self.terms.copy()
        info['throw_profile'] = self.profile.copy()
        return obs,reward,term,trunc,info
