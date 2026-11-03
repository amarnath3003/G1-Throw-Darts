"""Contact-fitted pickup baseline. Explicitly tuned, not learned from HaMeR.

Planning may pose a scratch MjData. Execution changes actuator commands only.
The object begins free on the table and is never teleported into the hand.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from env import PickupScene, ARM, HERE, geom_z_bounds
from retarget_arm import ArmIK, smoothstep

OUT = HERE/'research/pickup_attempt'
PAD_NAMES = ['L_thumb_distal', 'L_index_intermediate', 'L_middle_intermediate']


def torso_pairs(m):
    arm = [g for g in range(m.ngeom) if m.geom_contype[g] and
           m.body(m.geom_bodyid[g]).name.startswith(('left_shoulder_yaw', 'left_elbow', 'left_wrist', 'L_'))]
    torso = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name == 'torso_link']
    return [(a, b) for a in arm for b in torso]


def set_fingers(s, d, q):
    d.qpos[s.fq] = q
    m = s.model
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            a, b = m.eq_obj1id[e], m.eq_obj2id[e]
            d.qpos[m.jnt_qposadr[a]] = np.polynomial.polynomial.polyval(d.qpos[m.jnt_qposadr[b]], m.eq_data[e, :5])


class ContactFit:
    def __init__(self, s):
        self.s, self.m = s, s.model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = s.data.qpos
        self.pads = [self.m.body(n).id for n in PAD_NAMES]
        self.geoms = [self.m.geom(n+'_collision_0').id for n in PAD_NAMES]
        self.hand_geoms = [g for g in range(self.m.ngeom) if self.m.geom_contype[g] and
                           self.m.body(self.m.geom_bodyid[g]).name.startswith('L_')]
        self.bars = [g for g in s.dart_geoms if '_barrel_' in self.m.geom(g).name]
        self.nonbar = [g for g in s.dart_geoms if g not in self.bars]
        self.dart_pos = s.data.xpos[s.dart].copy()
        self.dart_rot = s.data.xmat[s.dart].reshape(3, 3).copy()
        # Recover actual pad points from the existing validated holding geometry.
        # This artificial object placement is ONLY in this scratch planning data.
        set_fingers(s, self.d, s.grasp['q_grasp'])
        mujoco.mj_forward(self.m, self.d)
        r = self.d.xmat[s.hand].reshape(3, 3)
        self.d.qpos[s.dq:s.dq+3] = self.d.xpos[s.hand]+r@np.array(s.grasp['dart_pos_hand'])
        local = Rotation.from_quat(np.array(s.grasp['dart_quat_hand'])[[1, 2, 3, 0]]).as_matrix()
        self.d.qpos[s.dq+3:s.dq+7] = Rotation.from_matrix(r@local).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(self.m, self.d)
        self.local_pads = []
        self.pad_axis = []
        for body, g in zip(self.pads, self.geoms):
            dist, line = self.nearest(g)
            self.local_pads.append(self.d.xmat[body].reshape(3, 3).T@(line[:3]-self.d.xpos[body]))
            self.pad_axis.append((self.d.xmat[s.dart].reshape(3, 3).T@(line[3:]-self.d.xpos[s.dart]))[0])
        self.local_pads = np.array(self.local_pads)
        self.pad_axis = np.array(self.pad_axis)
        self.pad_axis -= np.mean(self.pad_axis)
        self.d.qpos[:] = s.data.qpos
        self.armik = ArmIK(s)
        self.torso_pairs = torso_pairs(self.m)
        self.arm_geoms = {g for g in range(self.m.ngeom) if self.m.geom_contype[g] and
                          self.m.body(self.m.geom_bodyid[g]).name.startswith(('left_shoulder', 'left_elbow', 'left_wrist', 'L_'))}
        self.lo = np.r_[self.m.jnt_range[s.arm_ids, 0]+.06, [.05, .02, .05, .05]]
        self.hi = np.r_[self.m.jnt_range[s.arm_ids, 1]-.06, [1.295, .495, 1.4, 1.4]]
        self.finger_prior = np.array(s.grasp['q_grasp'][:4])

    def nearest(self, g):
        values = [mujoco.mj_geomDistance(self.m, self.d, g, b, .5, None) for b in self.bars]
        b = self.bars[int(np.argmin(values))]
        line = np.zeros(6)
        distance = mujoco.mj_geomDistance(self.m, self.d, g, b, .5, line)
        return distance, line

    def apply(self, x):
        self.d.qpos[self.s.aq] = x[:7]
        set_fingers(self.s, self.d, [*x[7:], 1.55, 1.55])
        mujoco.mj_forward(self.m, self.d)

    def assess(self, x):
        self.apply(x)
        contact = [float(self.nearest(g)[0]) for g in self.geoms]
        radial = []
        for g in self.geoms:
            _, line = self.nearest(g)
            v = (self.dart_rot.T@(line[3:]-self.dart_pos))[1:]
            radial.append(v/max(np.linalg.norm(v), 1e-9))
        opposition = [float(np.dot(radial[0], v)) for v in radial[1:]]
        heights = [geom_z_bounds(self.m, self.d, g)[0]-self.s.config['table_top_z'] for g in self.hand_geoms]
        unintended = []
        for c in self.d.contact:
            if c.dist < -.0005 and (c.geom1 in self.arm_geoms or c.geom2 in self.arm_geoms):
                if not (c.geom1 in self.s.dart_geoms or c.geom2 in self.s.dart_geoms):
                    unintended.append([int(c.geom1), int(c.geom2), float(c.dist)])
        return dict(pad_barrel_distances_m=contact, min_hand_table_height_m=min(heights),
                    thumb_finger_radial_dot=opposition,
                    unintended_penetrations=unintended, wrist_position=self.d.xpos[self.s.hand].tolist(),
                    wrist_rotation=self.d.xmat[self.s.hand].reshape(3, 3).tolist())

    def fit(self, refine=False):
        s, m, d = self.s, self.m, self.d
        baseline = json.loads((HERE/'research/setup_audit.json').read_text())[0]['terminal_grip_on_table']['joint_positions']
        seed_arm = np.array([baseline[n] for n in ARM])
        records = []
        best = None
        previous = json.loads((OUT/'contact_fit.json').read_text())['candidates'] if refine else None
        # Sideways pad opposition preserves access above the table. Search both
        # mirrored approach branches and modest rotations around the barrel.
        for k, angle in enumerate([0, 180, 30, 150, -30, 210]):
            radial = self.dart_rot @ np.array([0, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            goals = np.array([self.dart_pos+self.dart_rot[:, 0]*(a-.006)+sign*.00358*radial
                              for a, sign in zip(self.pad_axis, [1, -1, -1])])
            seed = np.r_[seed_arm if k % 2 == 0 else [.5, .3, .7, -.4, 1., -.5, -.2], self.finger_prior]

            def loss(x):
                self.apply(x)
                points = np.array([d.xpos[b]+d.xmat[b].reshape(3, 3)@p for b, p in zip(self.pads, self.local_pads)])
                heights = np.array([geom_z_bounds(m, d, g)[0] for g in self.hand_geoms])
                return np.r_[300*(points-goals).ravel(),
                             500*np.maximum(s.config['table_top_z']+.0007-heights, 0),
                             700*np.maximum(.004-self.armik.distances(self.torso_pairs, d), 0),
                             .06*(x[7:]-self.finger_prior),
                             .03*(x[:7]-self.armik.preferred)]
            if previous is not None:
                fitted = np.array(previous[k]['q'])
            else:
                fit = least_squares(loss, np.clip(seed, self.lo+1e-5, self.hi-1e-5),
                                    bounds=(self.lo, self.hi), max_nfev=220, diff_step=.001,
                                    ftol=1e-7, xtol=1e-8)
                fitted = fit.x
            def surface_loss(x):
                self.apply(x)
                points = np.array([d.xpos[b]+d.xmat[b].reshape(3, 3)@p for b, p in zip(self.pads, self.local_pads)])
                heights = np.array([geom_z_bounds(m, d, g)[0] for g in self.hand_geoms])
                nearest = [self.nearest(g) for g in self.geoms]
                distances = np.array([v[0] for v in nearest])
                radial_errors = []
                for (_, line), sign in zip(nearest, [1, -1, -1]):
                    v = (self.dart_rot.T@(line[3:]-self.dart_pos))[1:]
                    desired = sign*np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                    radial_errors.extend(v/max(np.linalg.norm(v), 1e-9)-desired)
                self_dist = np.array([mujoco.mj_geomDistance(m, d, self.geoms[0], g, .5, None) for g in self.geoms[1:]])
                return np.r_[500*(distances-.00005), 25*(points-goals).ravel(),
                             1.5*np.array(radial_errors),
                             700*np.maximum(s.config['table_top_z']+.0004-heights, 0),
                             1000*np.maximum(.003-self_dist, 0),
                             700*np.maximum(.004-self.armik.distances(self.torso_pairs, d), 0),
                             .10*(x[7:]-self.finger_prior), .02*(x[:7]-self.armik.preferred)]
            fit = least_squares(surface_loss, fitted, bounds=(self.lo, self.hi),
                                max_nfev=200, diff_step=.001, ftol=1e-6, xtol=1e-7)
            assessment = self.assess(fit.x)
            distances = np.array(assessment['pad_barrel_distances_m'])
            score = float(np.linalg.norm(distances)*1000+max(-assessment['min_hand_table_height_m'], 0)*10000+
                          sum(-c[2] for c in assessment['unintended_penetrations'])*10000+
                          sum(max(dot+.8, 0) for dot in assessment['thumb_finger_radial_dot'])*5)
            rec = dict(candidate=k, angle_deg=angle, score=score, q=fit.x.tolist(), **assessment)
            records.append(rec)
            print(f'candidate {k} score={score:.2f} pads={np.round(distances*1000,2)} mm table={assessment["min_hand_table_height_m"]*1000:.2f} mm', flush=True)
            if best is None or score < best['score']:
                best = rec
        (OUT/'contact_fit.json').write_text(json.dumps(dict(provenance='contact geometry optimization, not HaMeR finger retargeting',
                                                          local_pad_points=self.local_pads.tolist(), candidates=records, selected=best), indent=2)+'\n')
        return best


def wrist_ik(s, d, pos, rot, seed):
    lo, hi = s.model.jnt_range[s.arm_ids].T
    pairs = torso_pairs(s.model)
    def residual(q):
        d.qpos[s.aq] = q
        mujoco.mj_forward(s.model, d)
        return np.r_[80*(d.xpos[s.hand]-pos), 3*Rotation.from_matrix(rot.T@d.xmat[s.hand].reshape(3, 3)).as_rotvec(),
                     600*np.maximum(.004-np.array([mujoco.mj_geomDistance(s.model, d, a, b, 1., None) for a, b in pairs]), 0),
                     .015*(q-seed)]
    fit = least_squares(residual, np.clip(seed, lo+.06, hi-.06), bounds=(lo+.05, hi-.05),
                        max_nfev=100, ftol=1e-8)
    residual(fit.x)
    return fit.x


def make_path(s, candidate, hover=.012, squeeze=.025, lift_delta=(.02, 0, .10)):
    m = s.model
    d = mujoco.MjData(m); d.qpos[:] = s.data.qpos
    closed = np.r_[candidate['q'][7:], 1.55, 1.55]
    preshape = closed+np.array([0, .06, -.15, -.15, 0, 0])
    squeeze_q = closed+np.array([0, -squeeze, squeeze, squeeze, 0, 0])
    wrist = np.array(candidate['wrist_position'])
    rot = np.array(candidate['wrist_rotation'])
    high = wrist+np.array([0, 0, .09])
    seed = np.array(candidate['q'][:7])
    set_fingers(s, d, preshape)
    highq = wrist_ik(s, d, high, rot, seed)
    times = np.arange(0, 17.0001, .05)
    commands, errors, clearances = [], [], []
    for t in times:
        if t <= 4:
            u = smoothstep(t/4)
            aq = (1-u)*s.data.qpos[s.aq]+u*highq
            fq = (1-u)*s.data.qpos[s.fq]+u*preshape
            desired = None
        elif t <= 7:
            u = smoothstep((t-4)/3)
            desired = (1-u)*high+u*(wrist+[0, 0, hover])
            fq = preshape
        elif t <= 9:
            u = smoothstep((t-7)/2)
            desired = wrist+[0, 0, hover*(1-u)]
            fq = (1-u)*preshape+u*closed
        elif t <= 11:
            desired = wrist
            fq = closed+smoothstep((t-9)/1.0)*(squeeze_q-closed)
        else:
            u = smoothstep((t-11)/3)
            desired = wrist+u*np.asarray(lift_delta)
            fq = squeeze_q
        set_fingers(s, d, fq)
        if desired is not None:
            aq = wrist_ik(s, d, desired, rot, seed)
            errors.append(float(np.linalg.norm(d.xpos[s.hand]-desired)))
        d.qpos[s.aq] = aq
        mujoco.mj_forward(m, d)
        seed = aq
        clearances.append(min(geom_z_bounds(m, d, g)[0]-s.config['table_top_z'] for g in range(m.ngeom)
                              if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('L_')))
        commands.append(np.r_[aq, fq])
    return times, np.array(commands), dict(max_wrist_ik_error_m=max(errors), min_reference_hand_table_height_m=min(clearances))


def air_model(s):
    sys.path.insert(0, str(HERE.parent))
    from dart_model import DartDynamics, properties, load_profile
    for filename in ('dart_model.py', 'dart_profile.json'):
        assert hashlib.sha256((HERE.parent/filename).read_bytes()).hexdigest() == s.config['source_sha256'][filename]
    expected = np.asarray(properties(load_profile())['inertia_kg_m2'])
    assert np.allclose(s.model.body_inertia[s.dart], expected, rtol=5e-6, atol=0)
    s.model.body_inertia[s.dart] = expected
    mujoco.mj_setConst(s.model, mujoco.MjData(s.model))
    mujoco.mj_forward(s.model, s.data)
    return DartDynamics(s.model)


def reference_commands(s, times, path, feedback=True, contact_trigger=True, hold_force_targets=None):
    """Yield baseline commands against live physics, without changing execution state.

    The caller advances physics between yields. Both the baseline and residual
    environment use this controller so zero residual has a meaningful reference.
    """
    m, d = s.model, s.data
    va = m.jnt_dofadr[s.arm_ids]
    finger_joints = [m.joint(n).id for n in ('L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint',
                                           'L_index_proximal_joint', 'L_middle_proximal_joint', 'L_ring_proximal_joint', 'L_pinky_proximal_joint')]
    flo, fhi = m.jnt_range[finger_joints].T
    finger_command = d.qpos[s.fq].copy()
    pad_ids = [m.body(n).id for n in PAD_NAMES]
    start = d.time
    initial_dart = d.xpos[s.dart].copy(); initial_axis = d.xmat[s.dart].reshape(3, 3)[:, 0].copy()
    reference_data = mujoco.MjData(m); reference_data.qpos[:] = d.qpos
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    arm_correction = np.zeros(7); correction_goal = np.zeros(7); force_offset = np.zeros(3)
    next_feedback = 8.
    grasp_time = None; ready_duration = 0.; held_fingers = None
    held_force_offset = None
    contact_force = np.zeros(6)
    while True:
        t = d.time-start
        motion_t = t if grasp_time is None else 11+(t-grasp_time)
        if hold_force_targets is not None and grasp_time is None:
            motion_t = min(motion_t, 10.9)
        target = np.array([np.interp(motion_t, times, path[:, j]) for j in range(13)])
        if feedback and t >= 8:
            if t >= next_feedback and (t < 11 or hold_force_targets is not None) and grasp_time is None:
                # Privileged simulator object-pose feedback, explicitly not a
                # vision estimator. Ignore barrel axial roll; track axis tilt.
                current_axis = d.xmat[s.dart].reshape(3, 3)[:, 0].copy()
                cross = np.cross(initial_axis, current_axis)
                angle = np.arctan2(np.linalg.norm(cross), np.dot(initial_axis, current_axis))
                delta = Rotation.from_rotvec(cross/max(np.linalg.norm(cross), 1e-12)*angle).as_matrix()
                reference_data.qpos[s.aq] = target[:7]
                mujoco.mj_forward(m, reference_data)
                desired_pos = d.xpos[s.dart]+delta@(reference_data.xpos[s.hand]-initial_dart)
                desired_rot = delta@reference_data.xmat[s.hand].reshape(3, 3)
                for _ in range(5):
                    reference_data.qpos[s.aq] = target[:7]+correction_goal
                    mujoco.mj_forward(m, reference_data)
                    mujoco.mj_jacBody(m, reference_data, jacp, jacr, s.hand)
                    error = np.r_[desired_pos-reference_data.xpos[s.hand],
                                  Rotation.from_matrix(desired_rot@reference_data.xmat[s.hand].reshape(3, 3).T).as_rotvec()]
                    jac = np.vstack([jacp[:, va], jacr[:, va]])
                    change = jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-4, error)
                    correction_goal = np.clip(correction_goal+np.clip(change, -.03, .03), -.25, .25)
                next_feedback += .05
            arm_correction += np.clip(correction_goal-arm_correction, -.25*m.opt.timestep, .25*m.opt.timestep)
            target[:7] += arm_correction
            if t >= 9:
                measured = np.zeros(3)
                for ci, c in enumerate(d.contact):
                    b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
                    if s.dart in (b1, b2):
                        other = b2 if b1 == s.dart else b1
                        if other in pad_ids:
                            mujoco.mj_contactForce(m, d, ci, contact_force)
                            measured[pad_ids.index(other)] += contact_force[0]
                desired_force = np.full(3, .8) if hold_force_targets is None else np.asarray(hold_force_targets)
                if (t < 11 and grasp_time is None) or hold_force_targets is not None:
                    force_offset = np.clip(force_offset+(desired_force-measured)*.04*m.opt.timestep, -.04, .10)
                target[8:11] += force_offset*np.array([-1, 1, 1])
                if contact_trigger and grasp_time is None and (t < 11 or hold_force_targets is not None):
                    supported_grip = measured[0] > .3 and sum(measured[1:]) > .4
                    if hold_force_targets is not None:
                        supported_grip = bool(np.all(measured > .65*desired_force))
                    ready_duration = ready_duration+m.opt.timestep if supported_grip else 0.
                    minimum_duration = .12 if hold_force_targets is None else .25
                    if ready_duration >= minimum_duration:
                        grasp_time = float(t)
                        held_fingers = finger_command.copy()
                        held_force_offset = force_offset.copy()
                        correction_goal = arm_correction.copy()
            if grasp_time is not None:
                target[7:] = held_fingers
                if hold_force_targets is not None:
                    target[8:11] += (force_offset-held_force_offset)*np.array([-1, 1, 1])
        finger_command += np.clip(np.clip(target[7:], flo, fhi)-finger_command,
                                  -m.opt.timestep*np.array([.5, .5/2.4, .5, .5, .5, .5]),
                                  m.opt.timestep*np.array([.5, .5/2.4, .5, .5, .5, .5]))
        target[7:] = finger_command
        ahead = np.array([np.interp(min(motion_t+.002, times[-1]), times, path[:, j]) for j in range(7)])
        behind = np.array([np.interp(max(motion_t-.002, 0), times, path[:, j]) for j in range(7)])
        velocity = np.zeros(m.nv); velocity[va] = (ahead-behind)/.004
        controller_state = np.r_[arm_correction, correction_goal, force_offset,
                                 ready_duration, 0. if grasp_time is None else t-grasp_time,
                                 np.zeros(6) if held_fingers is None else held_fingers,
                                 np.zeros(3) if held_force_offset is None else held_force_offset]
        yield target, velocity, grasp_time, controller_state


def run_trial(s, candidate, name, squeeze=.025, render=False, feedback=False, contact_trigger=False,
              command_path=None, trial_directory=None, controller_label=None, hold_force_targets=None):
    m, d = s.model, s.data
    air = air_model(s)
    times, path, planning = make_path(s, candidate, squeeze=squeeze) if command_path is None else command_path
    trial_dir = OUT/name if trial_directory is None else Path(trial_directory)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir/'candidate.json').write_text(json.dumps(candidate, indent=2)+'\n')
    np.savez_compressed(trial_dir/'commands.npz', time_s=times, arm_and_finger_q=path)
    if planning['max_wrist_ik_error_m'] > .01:
        (trial_dir/'report.json').write_text(json.dumps(dict(name=name, stage='planning', physics_executed=False,
            pickup_success=False, reason='Wrist target unreachable within 10 mm gate', planning=planning), indent=2)+'\n')
        raise RuntimeError('Wrist target unreachable')
    pad_ids = [m.body(n).id for n in PAD_NAMES]
    left_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] and
                  m.body(m.geom_bodyid[g]).name.startswith(('L_', 'left_wrist', 'left_elbow', 'left_shoulder'))}
    start = d.time
    initial_com = d.xipos[s.dart].copy()
    history, logs, video_time = [], [], []
    next_sample = 0.
    maximum_penetration = 0.
    abort = None
    contact_force = np.zeros(6)
    touch_peaks = {}
    arm_error = 0.
    supported_duration = 0.
    maximum_supported_duration = 0.
    controller = reference_commands(s, times, path, feedback, contact_trigger, hold_force_targets)
    while d.time-start <= times[-1]:
        t = d.time-start
        target, velocity, grasp_time, _ = next(controller)
        s.target[s.aq] = target[:7]
        s.target[s.fq] = target[7:]
        torque = s.kp*(s.target[s.rq]-d.qpos[s.rq])+s.kd*(velocity[s.rv]-d.qvel[s.rv])+d.qfrc_bias[s.rv]
        d.ctrl[:] = np.clip(torque, -s.limit, s.limit)
        air.apply(d)
        mujoco.mj_step(m, d)
        arm_error = max(arm_error, float(np.max(np.abs(d.qpos[s.aq]-target[:7]))))
        for c in d.contact:
            if c.geom1 in left_geoms or c.geom2 in left_geoms:
                maximum_penetration = max(maximum_penetration, -float(c.dist))
                if c.dist < -.002:
                    abort = f'Arm/hand penetration exceeds 2 mm at {t:.3f}s: {int(c.geom1)}/{int(c.geom2)}'
        support_forces = np.zeros(3); supported_table_touch = False
        for ci, c in enumerate(d.contact):
            b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
            if s.dart in (b1, b2):
                other = b2 if b1 == s.dart else b1
                mujoco.mj_contactForce(m, d, ci, contact_force)
                key = m.body(other).name
                touch_peaks[key] = max(touch_peaks.get(key, 0.), float(contact_force[0]))
                if other in pad_ids:
                    support_forces[pad_ids.index(other)] += contact_force[0]
                if m.geom('table_top').id in (c.geom1, c.geom2):
                    supported_table_touch = True
        lowest_now = min(geom_z_bounds(m, d, g)[0] for g in s.dart_geoms)-s.config['table_top_z']
        supported = lowest_now > .03 and not supported_table_touch and support_forces[0] > .05 and np.any(support_forces[1:] > .05)
        supported_duration = supported_duration+m.opt.timestep if supported else 0.
        maximum_supported_duration = max(maximum_supported_duration, supported_duration)
        if t+1e-9 >= next_sample or abort:
            mujoco.mj_forward(m, d)
            forces = np.zeros(3); table_touch = False; hand_table = 0.
            for ci, c in enumerate(d.contact):
                b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
                mujoco.mj_contactForce(m, d, ci, contact_force)
                if s.dart in (b1, b2):
                    other = b2 if b1 == s.dart else b1
                    if other in pad_ids:
                        forces[pad_ids.index(other)] += contact_force[0]
                    if m.geom('table_top').id in (c.geom1, c.geom2) and contact_force[0] > .001:
                        table_touch = True
                if m.geom('table_top').id in (c.geom1, c.geom2) and (c.geom1 in left_geoms or c.geom2 in left_geoms):
                    hand_table += contact_force[0]
            lowest = min(geom_z_bounds(m, d, g)[0] for g in s.dart_geoms)-s.config['table_top_z']
            history.append(d.qpos.copy()); video_time.append(t)
            logs.append(dict(time_s=float(t), pad_force_N=forces.tolist(), dart_table_contact=table_touch,
                             dart_clearance_m=lowest, dart_com_height_gain_m=float(d.xipos[s.dart, 2]-initial_com[2]),
                             hand_table_force_N=float(hand_table),
                             object_hand_relative=(d.xmat[s.hand].reshape(3, 3).T@(d.xpos[s.dart]-d.xpos[s.hand])).tolist()))
            next_sample += 1/30
        if abort:
            break
    success = bool(not abort and supported_duration >= 3. - 1e-9)
    result = dict(name=name, controller=controller_label or 'geometry-fitted baseline; NOT HaMeR finger retargeting or learned pickup',
                  candidate=candidate['candidate'], timestep_s=m.opt.timestep, squeeze_rad=squeeze,
                  planning=planning, max_arm_tracking_error_rad=arm_error, abort_reason=abort,
                  max_arm_hand_penetration_m=maximum_penetration, pickup_success=success,
                  max_dart_com_height_gain_m=max(r['dart_com_height_gain_m'] for r in logs),
                  final_dart_com_height_gain_m=logs[-1]['dart_com_height_gain_m'],
                  final_pad_force_N=logs[-1]['pad_force_N'], final_dart_clearance_m=logs[-1]['dart_clearance_m'],
                  dart_contact_peak_normal_N=touch_peaks, model_sha256=s.config['model_sha256'],
                  object_pose_feedback='MuJoCo ground-truth axis/position during acquisition' if feedback else 'none; fixed planned pose',
                  contact_triggered_lift=contact_trigger, grasp_trigger_time_s=grasp_time,
                  continuous_supported_hold_s=supported_duration, maximum_supported_hold_s=maximum_supported_duration,
                  hold_force_targets_N=None if hold_force_targets is None else np.asarray(hold_force_targets).tolist(),
                  logs=logs)
    np.savez_compressed(trial_dir/'rollout.npz', qpos=history, time_s=video_time)
    (trial_dir/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'logs'}, indent=2), flush=True)
    if render:
        render_trial(s, trial_dir, np.array(history), np.array(video_time), success, grasp_time, controller_label)
    return result


def render_trial(s, directory, states, times, success, grasp_time=None, controller_label=None):
    import cv2
    m = s.model; d = mujoco.MjData(m)
    cameras = []
    for lookat, distance, azimuth, elevation in [([.4, .18, 1.], 1.65, 135, -18), ([*s.config['dart_xy'], .86], .48, 250, -23)]:
        c = mujoco.MjvCamera(); c.lookat[:] = lookat
        c.distance = distance; c.azimuth = azimuth; c.elevation = elevation
        cameras.append(c)
    target = directory/'pickup_attempt.mp4'
    proc = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x480',
                             '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
    with mujoco.Renderer(m, height=480, width=640) as renderer:
        for i, (state, t) in enumerate(zip(states, times)):
            d.qpos[:] = state; mujoco.mj_forward(m, d)
            views = []
            for c in cameras:
                renderer.update_scene(d, camera=c); views.append(renderer.render().copy())
            frame = np.hstack(views)
            cv2.rectangle(frame, (0, 0), (1280, 55), (22, 25, 32), -1)
            lift_start = 11 if grasp_time is None else grasp_time
            phase = 'Approach' if t < 7 else ('Acquire / squeeze' if t < lift_start else 'Lift attempt / hold')
            cv2.putText(frame, f'G1 physical pickup attempt | {phase} | {t:.1f}s', (16, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, (controller_label or 'Contact-fitted controller (tuned); not HaMeR finger motion')+' | '+('LIFT PASSED' if success else 'PICKUP NOT VALIDATED'),
                        (16, 46), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 205, 120), 1, cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
            if i % 90 == 0 or i == len(states)-1:
                cv2.imwrite(str(directory/f'preview_{i:04d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('Video encoding failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', action='store_true')
    p.add_argument('--refine', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--candidate', type=int)
    p.add_argument('--candidate-file', type=Path)
    p.add_argument('--squeeze', type=float, default=.025)
    p.add_argument('--name', default='trial_01')
    p.add_argument('--timestep', type=float, default=.0005)
    p.add_argument('--render', action='store_true')
    p.add_argument('--feedback', action='store_true')
    p.add_argument('--contact-trigger', action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = PickupScene(args.timestep)
    if args.fit or args.refine:
        ContactFit(s).fit(args.refine)
    if args.run:
        fitted = json.loads((OUT/'contact_fit.json').read_text())
        candidate = fitted['selected'] if args.candidate is None else fitted['candidates'][args.candidate]
        if args.candidate_file:
            candidate = json.loads(args.candidate_file.read_text())
        run_trial(s, candidate, args.name, args.squeeze, args.render, args.feedback, args.contact_trigger)


if __name__ == '__main__':
    main()
