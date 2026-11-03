"""Residual pickup with live contacts, empty-hand resets and the pinned dart physics."""
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from env import ARM, FINGERS, HERE, PickupScene, geom_z_bounds
from retarget_arm import geom_bounds
from try_pickup import PAD_NAMES, air_model, reference_commands


def load_config(path=None):
    return json.loads(Path(path or HERE / 'configs/pickup_ppo.json').read_text())


class PickupResidualEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, randomize=True, timestep=None):
        super().__init__()
        self.config = dict(config or load_config())
        self.randomize = randomize
        self.scene = s = PickupScene(timestep or self.config['physics_dt'])
        m = s.model
        self.model, self.data = m, s.data
        self.air = air_model(s)
        reference = HERE / self.config['reference']
        with np.load(reference / 'commands.npz') as bundle:
            self.times = bundle['time_s'].copy()
            self.commands = bundle['arm_and_finger_q'].copy()
        self.reset_reference = json.loads((reference / 'reset.json').read_text())
        report = json.loads((reference / 'report.json').read_text())
        self.force_targets = np.array(report['hold_force_targets_N'])
        self.indices = np.r_[s.aq, s.fq]
        joints = [m.joint(name).id for name in ARM + FINGERS]
        self.dofs = m.jnt_dofadr[joints]
        self.low, self.high = m.jnt_range[joints].T
        self.pad_ids = [m.body(name).id for name in PAD_NAMES]
        self.left_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] and
                           m.body(m.geom_bodyid[g]).name.startswith(
                               ('L_', 'left_wrist', 'left_elbow', 'left_shoulder'))}
        self.table = m.geom('table_top').id
        self.scale = np.array(self.config['residual_scale_rad'])
        self.rate = np.array(self.config['target_rate_rad_s'])
        self.dt = self.config['control_dt']
        self.substeps = round(self.dt / m.opt.timestep)
        if not np.isclose(self.substeps * m.opt.timestep, self.dt):
            raise ValueError('Control period must be an integer number of physics steps')
        self.action_space = gym.spaces.Box(-1., 1., shape=(13,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        observation, _ = self.reset(seed=0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=observation.shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s, m, d = self.scene, self.model, self.data
        varied = self.randomize and self.np_random.random() >= self.config['nominal_probability']
        jitter = self.config['position_jitter_m'] if varied else 0.
        yaw = self.np_random.uniform(-self.config['yaw_jitter_deg'], self.config['yaw_jitter_deg']) if varied else 0.
        xy = np.array(self.reset_reference['dart_xy']) + self.np_random.uniform(-jitter, jitter, 2)
        rotation = Rotation.from_quat(np.array(self.reset_reference['dart_quat_wxyz'])[[1, 2, 3, 0]])
        quaternion = (Rotation.from_euler('z', yaw, degrees=True) * rotation).as_quat()[[3, 0, 1, 2]]
        s.config['dart_xy'] = xy.tolist()
        s.config['dart_quat_wxyz'] = quaternion.tolist()
        s.reset()
        # Small yaw changes can leave the flights rocking after the initial settle.
        # Wait through real physics; never clear velocity or reposition the dart.
        for _ in range(round(3. / m.opt.timestep)):
            if (np.linalg.norm(d.qvel[s.dv:s.dv+3]) < .002 and
                    np.linalg.norm(d.qvel[s.dv+3:s.dv+6]) < .02):
                break
            self.air.apply(d)
            s.step()
        bounds = np.array([geom_bounds(m, d, geom) for geom in s.dart_geoms])
        center, half = np.array(s.config['table_center'])[:2], np.array(s.config['table_half_size'])[:2]
        if (np.any(bounds[:, 0, :2].min(axis=0) < center-half+.005) or
                np.any(bounds[:, 1, :2].max(axis=0) > center+half-.005)):
            raise ValueError('Reset rejected: complete dart footprint must fit on table')
        if (np.linalg.norm(d.qvel[s.dv:s.dv+3]) >= .002 or
                np.linalg.norm(d.qvel[s.dv+3:s.dv+6]) >= .02):
            raise ValueError('Reset dart did not physically settle')
        self.start = d.time
        self.elapsed = 0.
        self.initial_com_z = float(d.xipos[s.dart, 2])
        self.controller = reference_commands(s, self.times, self.commands, hold_force_targets=self.force_targets)
        self.nominal = self.commands[0].copy()
        self.command = d.qpos[self.indices].copy()
        self.controller_state = np.zeros(28)
        self.grasp_time = None
        self.previous_action = np.zeros(13)
        self.hold = 0.
        self.max_hold = 0.
        self.max_lift = 0.
        self.was_lifted = False
        self.reason = None
        self.finished = False
        self.metrics = self.measure()
        if np.any(self.metrics['forces'] > .001) or not self.metrics['table_touch']:
            raise ValueError('Reset must start with empty hand and tabletop dart')
        self.previous_relative = self.relative_position()
        return self.observe(), {'reset_xy': xy.tolist(), 'reset_yaw_delta_deg': float(yaw)}

    def relative_position(self):
        s, d = self.scene, self.data
        return d.xmat[s.hand].reshape(3, 3).T @ (d.xpos[s.dart] - d.xpos[s.hand])

    def measure(self):
        s, m, d = self.scene, self.model, self.data
        forces = np.zeros(3)
        table_touch = False
        unwanted = 0.
        penetration = 0.
        normals = np.zeros((3, 3))
        force = np.zeros(6)
        for index, contact in enumerate(d.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = m.geom_bodyid[[a, b]]
            touches_arm = a in self.left_geoms or b in self.left_geoms
            if not touches_arm and s.dart not in (body_a, body_b):
                continue
            mujoco.mj_contactForce(m, d, index, force)
            if touches_arm:
                penetration = max(penetration, -float(contact.dist))
                if s.dart not in (body_a, body_b):
                    unwanted += max(0., force[0])
            if s.dart in (body_a, body_b):
                table_touch |= self.table in (a, b)
                other = body_b if body_a == s.dart else body_a
                if other in self.pad_ids:
                    pad = self.pad_ids.index(other)
                    forces[pad] += max(0., force[0])
                    normals[pad] += contact.frame[:3] * force[0] * (1 if body_b == s.dart else -1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        opposition = max(0., float(-np.min(normals[1:] @ normals[0])))
        clearance = min(geom_z_bounds(m, d, g)[0] for g in s.dart_geoms) - s.config['table_top_z']
        threshold = self.config['support_force_N']
        supported = bool(forces[0] > threshold and np.any(forces[1:] > threshold))
        qualifying = supported and clearance > self.config['success_clearance_m'] and not table_touch
        return dict(forces=forces, table_touch=bool(table_touch), clearance=float(clearance),
                    supported=supported, qualifying=qualifying, opposition=opposition,
                    unwanted=float(unwanted), penetration=penetration)

    def observe(self):
        s, d = self.scene, self.data
        hand_rotation = d.xmat[s.hand].reshape(3, 3)
        relative_rotation = hand_rotation.T @ d.xmat[s.dart].reshape(3, 3)
        motion_time = min(self.elapsed, 10.9) if self.grasp_time is None else 11+self.elapsed-self.grasp_time
        phase = np.array([self.elapsed < 7, 7 <= self.elapsed and self.grasp_time is None,
                          self.grasp_time is not None], dtype=float)
        observation = np.r_[d.qpos[self.indices], d.qvel[self.dofs], self.relative_position(),
                            relative_rotation[:, :2].ravel(), d.qvel[s.dv:s.dv+6],
                            d.xpos[s.dart] - np.array([*s.config['dart_xy'], s.config['table_top_z']]),
                            self.metrics['forces'], self.nominal-d.qpos[self.indices],
                            self.command-d.qpos[self.indices], self.previous_action,
                            phase, self.elapsed/18., motion_time/17., self.hold/3.,
                            self.metrics['clearance'], self.controller_state]
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError('Nonfinite pickup observation')
        return observation.astype(np.float32)

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset required after episode end')
        action = np.asarray(action, dtype=float)
        if action.shape != (13,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected 13 finite residual actions')
        action = np.clip(action, -1., 1.)
        # Keep exploration from producing a new motor offset every control tick.
        smoothing_time = self.config.get('residual_smoothing_seconds', 0.)
        if smoothing_time > 0.:
            blend = -np.expm1(-self.dt / smoothing_time)
            action = self.previous_action + blend * (action - self.previous_action)
        s, m, d = self.scene, self.model, self.data
        reward = 0.
        weights = self.config['reward']
        for _ in range(self.substeps):
            self.nominal, velocity, self.grasp_time, self.controller_state = next(self.controller)
            # Fade in before acquisition; every episode physically executes the approach.
            strength = float(np.clip((self.elapsed-6.)/1., 0., 1.))
            target = np.clip(self.nominal + strength*self.scale*action, self.low, self.high)
            self.command += np.clip(target-self.command, -self.rate*m.opt.timestep, self.rate*m.opt.timestep)
            s.target[self.indices] = self.command
            torque = s.kp*(s.target[s.rq]-d.qpos[s.rq]) + s.kd*(velocity[s.rv]-d.qvel[s.rv]) + d.qfrc_bias[s.rv]
            d.ctrl[:] = np.clip(torque, -s.limit, s.limit)
            self.air.apply(d)
            mujoco.mj_step(m, d)
            self.elapsed = float(d.time-self.start)
            self.metrics = metrics = self.measure()
            self.hold = self.hold+m.opt.timestep if metrics['qualifying'] else 0.
            self.max_hold = max(self.max_hold, self.hold)
            self.max_lift = max(self.max_lift, float(d.xipos[s.dart, 2]-self.initial_com_z))
            self.was_lifted |= metrics['qualifying']
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                raise FloatingPointError('Nonfinite pickup physics')
            if metrics['penetration'] > self.config['max_penetration_m']:
                self.reason = 'penetration'
            elif metrics['clearance'] < -.03:
                self.reason = 'off_table'
            elif self.was_lifted and metrics['table_touch']:
                self.reason = 'dropped'
            if self.reason or self.hold >= self.config['success_hold_s'] - 1e-9:
                break
        metrics = self.metrics
        relative = self.relative_position()
        slip = np.linalg.norm(relative-self.previous_relative)/self.dt if metrics['supported'] else 0.
        self.previous_relative = relative
        contact_quality = min(metrics['forces'][0]/.8, np.max(metrics['forces'][1:])/.5, 1.) * metrics['opposition']
        height = np.clip(metrics['clearance']/.10, 0., 1.) * metrics['supported']
        reach = np.exp(-np.linalg.norm(relative-np.array(s.grasp['dart_pos_hand']))/.05)
        rate_reward = (weights['reach']*reach + weights['opposing_contact']*contact_quality +
                       weights['supported_height']*height + weights['continuous_hold']*min(self.hold/3., 1.) -
                       weights['slip']*min(slip/.02, 5.) -
                       weights['excess_force']*np.maximum(metrics['forces']-3., 0.).sum() -
                       weights['unwanted_contact']*min(metrics['unwanted'], 30.) -
                       weights['residual']*np.mean(action**2) -
                       weights['action_change']*np.mean((action-self.previous_action)**2))
        reward += self.dt*rate_reward
        success = self.hold >= self.config['success_hold_s'] - 1e-9 and self.reason is None
        terminated = bool(success or self.reason)
        truncated = bool(self.elapsed >= self.config['episode_seconds']-1e-9 and not terminated)
        if success:
            reward += weights['success']
        elif self.reason:
            reward -= weights['failure']
        self.previous_action = action.copy()
        self.finished = terminated or truncated
        info = dict(is_success=bool(success), reason=self.reason or ('success' if success else 'timeout' if truncated else 'running'),
                    max_lift_m=self.max_lift, max_hold_s=self.max_hold,
                    clearance_m=metrics['clearance'], hold_s=self.hold, elapsed_s=self.elapsed,
                    grasp_trigger_s=self.grasp_time)
        return self.observe(), float(reward), terminated, truncated, info
