"""Episode-level residual PPO over smooth robot joint references.

The frozen legacy policy supplies feedback and the demonstrated swing. PPO
chooses coherent motion corrections once per throw, avoiding saturated noisy
per-step controls. All outcomes come from the original contact simulation.
"""
from pathlib import Path
import json
import cloudpickle
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from calibrated_env import CalibratedThrowEnv

PARAMETERS = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
              'wrist_pitch', 'waist_yaw', 'waist_pitch', 'release_delay']
JOINT_INDICES = np.array([0, 1, 2, 3, 5, 7, 8])
AMPLITUDES = np.array([.10, .08, .10, .15, .12, .10, .06])
RAMP_SECONDS = .12


class CorrectedCurve:
    def __init__(self, base, addresses, offsets):
        self.base, self.addresses, self.offsets = base, addresses, offsets

    def __call__(self, time, derivative=0):
        result = self.base(time, derivative).copy()
        u = np.clip((np.asarray(time)-2.1)/RAMP_SECONDS, 0., 1.)
        if derivative == 0:
            blend = u*u*(3-2*u)
        elif derivative == 1:
            blend = 6*u*(1-u)/RAMP_SECONDS
        else:
            raise ValueError('Only position/velocity references are supported')
        result[..., self.addresses] += np.asarray(blend)[..., None]*self.offsets
        return result


def precision_objective(info, parameters):
    """Actual endpoint reward; prediction and distance travelled earn nothing."""
    clean = info.get('release') is not None and info['outcome'] in ('board','environment_hit','timeout')
    board = info['reward_terms'].get('board_impact',0.) > 0
    if info['outcome'] == 'board' and not board:
        clean = False
    crossing = info.get('board_plane_crossing')
    error = info.get('impact_error_m') if board else (crossing['error_m'] if clean and crossing else None)
    if not clean:
        value = -40.
    elif error is None:
        value = -15.
    else:
        value = sum(w*np.exp(-(error/width)**2) for w,width in ((20.,.4),(40.,.12),(100.,.04)))
        value += 5. if board else -5.
        if info.get('is_success'):
            value += 100.
    return float(value-.02*np.mean(np.square(parameters)))


class ParameterThrowEnv(gym.Wrapper):
    def __init__(self, profile, **kwargs):
        if profile.get('controller') != 'parameter_ppo_v1':
            raise ValueError('Unsupported controller version')
        super().__init__(CalibratedThrowEnv(profile=profile['base_profile'], **kwargs))
        self.controller_profile = profile
        teacher = Path(profile['teacher_checkpoint'])
        if not teacher.is_absolute():
            teacher = Path(__file__).resolve().parents[1]/teacher
        self.teacher = PPO.load(str(teacher), device='cpu')
        with (teacher.parent/'normalize.pkl').open('rb') as handle:
            self.teacher_norm = cloudpickle.load(handle)
        self.action_space = gym.spaces.Box(-1.,1.,shape=(8,),dtype=np.float32)
        self.base_curve = self.env.curve
        self.transition_controls = []
        self._done = True
        # Check the entire allowed correction envelope against original robot
        # limits, including worst-case reference speed + legacy residual slew.
        samples = np.linspace(self.env.plan['start'],self.env.plan['end'],1001)
        addresses = self.env.qadr[JOINT_INDICES]
        q = self.base_curve(samples)[:,addresses]
        limits = self.env.m.jnt_range[self.env.jids[JOINT_INDICES]]
        u=np.clip((samples-2.1)/RAMP_SECONDS,0.,1.)
        envelope=(u*u*(3-2*u))[:,None]*AMPLITUDES
        if np.any(q-envelope < limits[:,0]+.04) or np.any(q+envelope > limits[:,1]-.04):
            raise ValueError('Correction envelope exceeds joint reference margins: '+str(np.minimum((q-envelope-(limits[:,0]+.04)).min(0),((limits[:,1]-.04)-q-envelope).min(0))))
        speed = np.abs(self.base_curve(samples,1)[:,addresses])+1.5*AMPLITUDES/RAMP_SECONDS
        speed += np.maximum(.06,self.env.slew[JOINT_INDICES])/.02
        if np.any(speed > self.env.reference_speed_limits[JOINT_INDICES]):
            raise ValueError('Correction envelope exceeds original joint speed limits')

    def __getattr__(self,name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.env,name)

    def reset(self, **kwargs):
        self.env.curve = self.base_curve
        self.transition_controls = []
        self._done = False
        self._obs, info = self.env.reset(**kwargs)
        return self._obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset before starting another throw')
        parameters = np.asarray(action,dtype=float)
        if parameters.shape != (8,) or not np.isfinite(parameters).all() or np.any(abs(parameters)>1.000001):
            raise ValueError('Invalid throw parameters')
        offsets = AMPLITUDES*parameters[:7]
        self.env.curve = CorrectedCurve(self.base_curve,self.env.qadr[JOINT_INDICES],offsets)
        controls=[]; traces=[]; done=False; total=0.
        # A negative delay retains the earliest legal opening. This boundary is
        # explicit and logged; no unphysical earlier release is introduced.
        delay = .025*max(0.,float(parameters[-1]))
        while not done:
            base_action,_ = self.teacher.predict(self.teacher_norm.normalize_obs(self._obs[None,:]),deterministic=True)
            base_action=base_action[0]
            if (self.env.scheduled_release_time is None and not self.env.commanded
                    and self.env.t>=self.env.release_earliest and base_action[-1]>.6):
                self.env.scheduled_release_time=self.env.t+delay
            traces.append(dict(time_s=self.env.t, teacher_action=base_action.tolist(),
                               joint_q=self.env.d.qpos[self.env.qadr].tolist(),
                               joint_qd=self.env.d.qvel[self.env.vadr].tolist()))
            self._obs,r,term,trunc,info=self.env.step(base_action)
            total+=r;controls.extend(self.env.transition_controls);done=term or trunc
        reward = precision_objective(info,parameters)
        correction=reward-total
        last=controls[-1];controls[-1]=(*last[:2],last[2]+correction,*last[3:])
        self.transition_controls=controls
        info['reward_terms']['parameter_objective_correction']=correction
        info['parameter_reward']=reward
        info['parameter_names']=PARAMETERS
        info['throw_parameters']=parameters.tolist()
        info['reference_offsets_rad']=offsets.tolist()
        info['release_delay_s']=delay
        info['teacher_trace']=traces
        info['physics_duration_s']=self.env.t
        info['controller']='parameter_ppo_v1'
        self._done=True
        # This is a completed full throw, including physics timeout outcomes.
        # It is never a time-limit truncated PPO transition with bootstrapping.
        return self._obs,reward,True,False,info
