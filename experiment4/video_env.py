"""Video to Throw PPO using Experiment 2's physical release and impact gates."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from env import DartThrowEnv
from inspire_grip import InspireGrip
from inspire_model import NAMES
from parameter_env import precision_objective

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
              'wrist_pitch', 'wrist_yaw', 'stroke_tempo', 'release_delay']
JOINT_INDICES = np.array([0, 1, 2, 3, 5, 6])
AMPLITUDES = np.array([.10, .08, .10, .15, .12, .10])


class VideoCurve:
    def __init__(self, reference, model, fixed):
        self.time = reference['time']
        self.frames = reference['source_frame']
        self.addresses = np.array([model.joint(str(n)).qposadr[0] for n in reference['joints']])
        self.base = PchipInterpolator(self.time, reference['q'], axis=0)
        self.fixed = fixed.copy()
        self.offsets = np.zeros(7)
        # Corrections begin after the second pullback has reached its peak.
        self.correction_start = float(np.interp(212, self.frames, self.time))
        self.ramp = .12

    def __call__(self, time, derivative=0):
        value = np.asarray(time)
        result = np.broadcast_to(self.fixed, (*value.shape, len(self.fixed))).copy()
        if derivative:
            result[...] = 0.
        u = np.clip((value-self.correction_start)/self.ramp, 0., 1.)
        blend = u*u*(3-2*u) if derivative == 0 else 6*u*(1-u)/self.ramp
        result[..., self.addresses] = self.base(value, derivative) + blend[..., None]*self.offsets
        return result


class VideoPhysics(DartThrowEnv):
    def __init__(self, profile, **kwargs):
        self.profile = profile
        path = Path(profile['reference'])
        self.reference_path = path if path.is_absolute() else ROOT/path
        if hashlib.sha256(self.reference_path.read_bytes()).hexdigest() != profile['reference_sha256']:
            raise ValueError('Video reference checksum mismatch')
        self.reference = np.load(self.reference_path)
        self.tempo = 1.
        kwargs.pop('variant', None)
        super().__init__(variant='arc_v7', **kwargs)
        self.shoulder_y = float(self.d.xpos[self.m.body('left_shoulder_pitch_link').id, 1])
        self.elbow_body = self.m.body('left_elbow_link').id
        self.head = next(i for i in range(self.m.ngeom)
                         if self.m.geom_contype[i] and self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                         and self.m.mesh(self.m.geom_dataid[i]).name == 'head_link')
        self.head_bounds = self.bounds(self.head, self.vertices(self.head))
        self.dart_shapes = [(i, self.vertices(i)) for i in np.flatnonzero(self.m.geom_bodyid == self.sim.dart)]
        urdf = ET.parse(ROOT/'assets/inspire/upstream_g1.urdf').getroot()
        self.speed_limits = np.array([float(urdf.find(f"joint[@name='{name}']/limit").get('velocity'))
                                      for name in self.names])

    def make_simulator(self):
        return InspireGrip(fixed_lower_body=True)

    def configure_reference(self):
        self.names = self.reference['joints'].tolist()
        for name, value in zip(self.names, self.reference['q'][0]):
            self.sim.config['arm_pose'][name] = float(value)
        for name in NAMES:
            right = 'R_'+name[2:]
            self.sim.config['arm_pose'][right] = float(self.d.qpos[self.m.joint(right).qposadr[0]])
        self.sim.reset()
        self.curve = VideoCurve(self.reference, self.m, self.d.qpos)
        self.plan = dict(start=0., end=float(self.reference['time'][-1]))
        self.source_release = float(np.interp(225, self.curve.frames, self.curve.time))
        self.tempo_start = self.curve.correction_start
        self.tempo_ramp = .12
        self.release_deadline = self.plan['end']+1.
        self.episode_deadline = self.plan['end']+2.
        self.reference_margin = .015

    def phase_at(self, elapsed):
        # Smoothly change playback speed only for the final forward stroke.
        elapsed = np.asarray(elapsed)
        u = np.clip((elapsed-self.tempo_start)/self.tempo_ramp, 0., 1.)
        integral = self.tempo_ramp*(u**3-.5*u**4)
        integral += np.maximum(elapsed-self.tempo_start-self.tempo_ramp, 0.)
        phase = elapsed+(self.tempo-1.)*integral
        rate = 1.+(self.tempo-1.)*u*u*(3.-2.*u)
        return np.minimum(phase, self.plan['end']), np.where(phase < self.plan['end'], rate, 0.)

    def reference_phase(self):
        phase, rate = self.phase_at(self.t)
        return float(phase), float(rate)

    def reset(self, **kwargs):
        self.curve.offsets[:] = 0.
        self.tempo = 1.
        self.motion_trace = []
        self.minimum_lane = float('inf')
        self.minimum_head_gap = float('inf')
        return super().reset(**kwargs)

    def vertices(self, geom):
        if self.m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            return None
        mesh = self.m.geom_dataid[geom]
        start = self.m.mesh_vertadr[mesh]
        return self.m.mesh_vert[start:start+self.m.mesh_vertnum[mesh]]

    def bounds(self, geom, vertices):
        if vertices is None:
            radius = self.m.geom_rbound[geom]
            return self.d.geom_xpos[geom]-radius, self.d.geom_xpos[geom]+radius
        points = vertices @ self.d.geom_xmat[geom].reshape(3, 3).T+self.d.geom_xpos[geom]
        return points.min(axis=0), points.max(axis=0)

    def motion_violation(self):
        phase, _ = self.reference_phase()
        wrist = self.d.xpos[self.sim.hand]
        lane = min(wrist[1], self.d.xpos[self.elbow_body, 1])-self.shoulder_y
        self.minimum_lane = min(self.minimum_lane, float(lane))
        self.motion_trace.append((phase, float(wrist[0])))
        if lane < 0.:
            return 'shoulder_lane_violation'
        low_head, high_head = self.head_bounds
        gap = min(float(np.max(np.maximum(low-high_head, low_head-high)))
                  for low, high in (self.bounds(g, v) for g, v in self.dart_shapes))
        self.minimum_head_gap = min(self.minimum_head_gap, gap)
        if gap <= .005:
            return 'dart_head_clearance'
        return None

    def swing_audit(self):
        trace = np.asarray(self.motion_trace)
        source = np.interp(trace[:, 0], self.curve.time, self.curve.frames)
        checks = []
        for begin, back, end in (((103, 119), (137, 154), (165, 179)),
                                  ((165, 179), (198, 212), (224, 236))):
            parts = [trace[(source >= low) & (source <= high), 1] for low, high in (begin, back, end)]
            if any(len(part) == 0 for part in parts):
                checks.append(dict(passed=False, reason='incomplete swing'))
                continue
            retreat = float(np.median(parts[0])-np.median(parts[1]))
            advance = float(np.max(parts[2])-np.median(parts[1]))
            checks.append(dict(passed=retreat >= .03 and advance >= .03, retreat_m=retreat, advance_m=advance))
        return checks


class VideoThrowEnv(gym.Wrapper):
    def __init__(self, profile, **kwargs):
        if profile['controller'] != 'video_to_throw_v1':
            raise ValueError('Unsupported video controller')
        super().__init__(VideoPhysics(profile, **kwargs))
        self.action_space = gym.spaces.Box(-1., 1., (8,), dtype=np.float32)
        self.transition_controls = []
        self._done = True

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.transition_controls = []
        self._done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._done:
            raise RuntimeError('Reset before starting another throw')
        parameters = np.asarray(action, dtype=float)
        if parameters.shape != (8,) or not np.isfinite(parameters).all() or np.any(abs(parameters) > 1.000001):
            raise ValueError('Invalid throw parameters')
        env = self.env
        env.curve.offsets[JOINT_INDICES] = AMPLITUDES*parameters[:6]
        env.tempo = float(2.**parameters[6])
        # Finger opening keeps its physical latency in seconds at every tempo.
        times = np.linspace(0., env.episode_deadline, 10001)
        phases, rates = env.phase_at(times)
        release_time = float(np.interp(env.source_release, phases, times))
        env.scheduled_release_time = release_time-.07+.08*parameters[7]
        speed = abs(env.curve(phases, 1)[:, env.qadr])*rates[:, None]
        if np.any(speed > env.speed_limits+1e-6):
            raise ValueError('Residual reference exceeds manufacturer velocity limits')
        total = 0.
        control = np.zeros(8)
        while True:
            start = env.t
            obs, reward, terminated, truncated, info = env.step(control)
            total += reward
            self.transition_controls.append((start, parameters.copy(), reward, env.sim.target.copy(),
                                             env.sim.velocity_target.copy(), env.d.ctrl.copy()))
            if terminated or truncated:
                break
        checks = env.swing_audit()
        valid_motion = all(check['passed'] for check in checks)
        info['physical_outcome'] = info['outcome']
        if not valid_motion:
            if info['outcome'] in ('board', 'environment_hit', 'timeout'):
                info['outcome'] = 'incomplete_two_swings'
            info['is_success'] = False
            info['physical_board_impact_reward'] = info['reward_terms'].get('board_impact', 0.)
            info['reward_terms']['board_impact'] = 0.
        objective = precision_objective(info, parameters)
        # The video baseline falls short of the board. Measured free-flight
        # proximity supplies a gradient until the existing endpoint objective
        # becomes informative. This term never declares a board hit or success.
        approach = 0.
        if (valid_motion and info.get('release') is not None
                and info['outcome'] in ('board', 'environment_hit', 'timeout')
                and info.get('closest_tip_distance_m') is not None):
            approach = 8./(1.+(info['closest_tip_distance_m']/.5)**2)
        reward = objective+approach
        correction = reward-total
        row = self.transition_controls[-1]
        self.transition_controls[-1] = (*row[:2], row[2]+correction, *row[3:])
        info['reward_terms']['video_objective_correction'] = reward-sum(info['reward_terms'].values())
        info.update(controller='video_to_throw_v1', parameter_names=PARAMETERS,
                    throw_parameters=parameters.tolist(), parameter_reward=reward,
                    precision_reward=objective, approach_reward=approach,
                    reference_offsets_rad=env.curve.offsets.tolist(), stroke_tempo=env.tempo,
                    scheduled_release_s=env.scheduled_release_time, swing_checks=checks,
                    min_shoulder_lane_m=env.minimum_lane, min_dart_head_gap_m=env.minimum_head_gap,
                    physics_duration_s=env.t, reference_sha256=env.profile['reference_sha256'])
        self._done = True
        return obs, reward, True, False, info
