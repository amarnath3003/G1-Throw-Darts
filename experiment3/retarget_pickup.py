"""Video closure guidance and contact-constrained acquisition in the pinned scene.

All fitting poses use scratch data. Execution uses bounded joint torques only.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

from env import HERE, ARM, FINGERS, PickupScene, geom_z_bounds
from try_pickup import ContactFit, set_fingers, wrist_ik, make_path, run_trial
from retarget_arm import geom_bounds


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def video_guidance(dataset, out):
    path = dataset / 'milestone2/4276/reference.npz'
    ref = np.load(path)
    valid = ref['usable_2d_reference'] & ~ref['temporal_jump_flags']
    ts = ref['timestamps_s']
    distance = np.mean(ref['thumb_to_fingertip_distances_model_m'][:, :2], axis=1)
    opened = float(np.median(distance[valid & (ref['phases'] == 'approach')]))
    closed = float(np.median(distance[valid & (ref['phases'] == 'lift')]))
    if opened - closed < .01:
        raise ValueError('Video does not resolve a reliable closure change')
    selection = valid & np.isin(ref['phases'], ['approach', 'acquisition', 'lift'])
    time = ts[selection]
    closure = np.clip((opened - distance[selection]) / (opened - closed), 0, 1)
    # A monotone fit avoids reopening on brief occlusion-induced excursions.
    from scipy.optimize import isotonic_regression
    closure = isotonic_regression(closure).x
    closure = (closure - closure[0]) / max(closure[-1] - closure[0], 1e-6)
    np.savez_compressed(out / 'video_guidance.npz', source_time_s=time, closure=closure)
    report = dict(clip='4276', reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_start_s=float(time[0]), source_end_s=float(time[-1]),
                  observed_open_distance_m=opened, observed_closed_distance_m=closed,
                  guidance='Normalized thumb/index/middle closure timing; monotone fit',
                  geometry='Robot/dart contact fit; video supplies closure progression only',
                  translation='Designed approach and 100 mm vertical lift',
                  depth_calibrated=False, contacts_measured=False, holdout_used=False)
    save_json(out / 'video_guidance.json', report)
    return time, closure


class AcquisitionFit(ContactFit):
    def __init__(self, scene):
        super().__init__(scene)
        s, m, d = self.s, self.m, self.d
        set_fingers(s, d, s.grasp['q_grasp'])
        mujoco.mj_forward(m, d)
        rotation = d.xmat[s.hand].reshape(3, 3)
        d.qpos[s.dq:s.dq+3] = d.xpos[s.hand] + rotation @ np.array(s.grasp['dart_pos_hand'])
        local = Rotation.from_quat(np.array(s.grasp['dart_quat_hand'])[[1, 2, 3, 0]]).as_matrix()
        d.qpos[s.dq+3:s.dq+7] = Rotation.from_matrix(rotation @ local).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(m, d)
        normals, dart_points, dart_normals = [], [], []
        for body, geom in zip(self.pads, self.geoms):
            distance, line = self.nearest(geom)
            if distance <= 0:
                raise ValueError('Pad reference must have separated surfaces')
            normals.append(d.xmat[body].reshape(3, 3).T @ ((line[3:] - line[:3]) / distance))
            dart_points.append(d.xmat[s.dart].reshape(3, 3).T @ (line[3:] - d.xpos[s.dart]))
            dart_normals.append(d.xmat[s.dart].reshape(3, 3).T @ ((line[3:] - line[:3]) / distance))
        self.pad_normals = np.array(normals)
        self.dart_points = np.array(dart_points)
        self.dart_normals = np.array(dart_normals)
        d.qpos[:] = s.data.qpos
        self.lo[7:] = [.05, .015, .05, .05]
        self.hi[7:] = [1.295, .495, 1.65, 1.65]

    def contact_geometry(self):
        points, normals, distances = [], [], []
        for geom in self.geoms:
            distance, line = self.nearest(geom)
            vector = line[3:] - line[:3]
            normals.append(vector / max(np.linalg.norm(vector), 1e-12) * np.sign(distance))
            points.append(line[3:])
            distances.append(distance)
        return np.array(points), np.array(normals), np.array(distances)

    def support(self):
        """Static friction-cone feasibility with bounded normal force and motor torque.

        Uses inscribed sliding/spinning cone rays and coupled joint Jacobians.
        This is a planning check, not evidence of a dynamic grasp.
        """
        s, m, d = self.s, self.m, self.d
        points, normals, _ = self.contact_geometry()
        if np.any(np.linalg.norm(normals, axis=1) < .9):
            return dict(feasible=False, required_normal_N=None, normal_dot=[1.],
                        note='Contact normal undefined at touching/intersecting geometry')
        wrench, torques = [], []
        coupling = np.zeros((m.nv, m.nu))
        for actuator, joint in enumerate(s.act_joints):
            coupling[m.jnt_dofadr[joint], actuator] = 1
        for eq in range(m.neq):
            if m.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
                child, parent = m.eq_obj1id[eq], m.eq_obj2id[eq]
                polynomial = np.polynomial.polynomial.polyder(m.eq_data[eq, :5])
                derivative = np.polynomial.polynomial.polyval(d.qpos[m.jnt_qposadr[parent]], polynomial)
                coupling[m.jnt_dofadr[child]] = derivative * coupling[m.jnt_dofadr[parent]]
        for point, normal, body, geom in zip(points, normals, self.pads, self.geoms):
            tangent = np.cross(normal, [1., 0., 0.])
            if np.linalg.norm(tangent) < .1:
                tangent = np.cross(normal, [0., 1., 0.])
            tangent /= np.linalg.norm(tangent)
            second = np.cross(normal, tangent)
            mu = min(float(m.geom_friction[geom, 0]), float(m.geom_friction[self.bars[0], 0]))
            jac = np.zeros((3, m.nv))
            jac_rotation = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, jac_rotation, point, body)
            for angle in np.arange(8) * (2 * np.pi / 8):
                force = normal + mu * (np.cos(angle) * tangent + np.sin(angle) * second)
                wrench.append(np.r_[force, np.cross(point - d.xipos[s.dart], force) / .03])
                torques.append(coupling.T @ jac.T @ force)
            spin = min(float(m.geom_friction[geom, 1]), float(m.geom_friction[self.bars[0], 1]))
            if min(m.geom_condim[geom], m.geom_condim[self.bars[0]]) < 4:
                spin = 0.
            for sign in [-1, 1]:
                moment = sign * spin * normal
                wrench.append(np.r_[normal, (np.cross(point - d.xipos[s.dart], normal) + moment) / .03])
                torques.append(coupling.T @ (jac.T @ normal + jac_rotation.T @ moment))
        wrench, torques = np.array(wrench).T, np.array(torques).T
        bias = coupling.T @ d.qfrc_bias
        count = len(self.pads)
        normal_sums = np.kron(np.eye(count), np.ones((1, 10)))
        result = linprog(np.ones(count * 10), A_ub=np.vstack([normal_sums, torques, -torques]),
                         b_ub=np.r_[np.full(count, 3.), s.limit - bias, s.limit + bias],
                         A_eq=wrench, b_eq=np.r_[-m.body_mass[s.dart] * m.opt.gravity, [0., 0., 0.]],
                         bounds=(0, None), method='highs')
        return dict(feasible=bool(result.success),
                    required_normal_N=(normal_sums @ result.x).tolist() if result.success else None,
                    normal_dot=(normals[1:] @ normals[0]).tolist(),
                    note='Static inscribed sliding/spinning cone; dynamic lift must be tested')

    def fit_candidates(self, out):
        s, m, d = self.s, self.m, self.d
        records = []
        local = Rotation.from_quat(np.array(s.grasp['dart_quat_hand'])[[1, 2, 3, 0]]).as_matrix()
        for candidate_index, angle_deg in enumerate(range(0, 360, 30)):
            rotation = self.dart_rot @ Rotation.from_rotvec([np.deg2rad(angle_deg), 0, 0]).as_matrix()
            desired_normals = self.dart_normals @ rotation.T
            goals = self.dart_pos + self.dart_points @ rotation.T
            wrist_rotation = rotation @ local.T
            wrist_position = self.dart_pos - wrist_rotation @ np.array(s.grasp['dart_pos_hand'])
            seed_arm = wrist_ik(s, d, wrist_position, wrist_rotation, self.armik.preferred)
            seed = np.r_[seed_arm, self.finger_prior]

            def residual(q):
                self.apply(q)
                points = np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ p
                                   for b, p in zip(self.pads, self.local_pads)])
                normals = np.array([d.xmat[b].reshape(3, 3) @ n
                                    for b, n in zip(self.pads, self.pad_normals)])
                heights = np.array([geom_z_bounds(m, d, g)[0] for g in self.hand_geoms])
                _, _, distances = self.contact_geometry()
                return np.r_[250 * (points - goals).ravel(),
                             .5 * (normals - desired_normals).ravel(),
                             1800 * (distances - .00008),
                             2500 * np.maximum(s.config['table_top_z'] + .0006 - heights, 0),
                             700 * np.maximum(.004 - self.armik.distances(self.torso_pairs, d), 0),
                             .04 * (q[7:] - self.finger_prior), .015 * (q[:7] - self.armik.preferred)]

            fit = least_squares(residual, np.clip(seed, self.lo + 1e-5, self.hi - 1e-5),
                                bounds=(self.lo, self.hi), max_nfev=220, diff_step=.0001)
            assessment = self.assess(fit.x)
            support = self.support()
            distances = np.array(assessment['pad_barrel_distances_m'])
            feasible = bool(np.max(np.abs(distances)) < .0008 and np.min(distances) >= -.0001
                            and assessment['min_hand_table_height_m'] >= .0002
                            and not assessment['unintended_penetrations']
                            and min(support['normal_dot']) < -.65 and support['feasible'])
            score = float(np.linalg.norm(distances) * 1000
                          + max(.0002 - assessment['min_hand_table_height_m'], 0) * 10000
                          + sum(max(dot + .8, 0) for dot in support['normal_dot']) * 5)
            record = dict(candidate=candidate_index, angle_deg=angle_deg, q=fit.x.tolist(), score=score,
                          **assessment, support=support, geometry_passed=feasible)
            records.append(record)
            save_json(out / 'candidates.json', records)
            print(json.dumps(dict(candidate=record['candidate'], passed=feasible, score=score,
                                  gaps_mm=(distances * 1000).round(3).tolist(),
                                  table_mm=round(assessment['min_hand_table_height_m'] * 1000, 3),
                                  support=support)), flush=True)
        passed = [record for record in records if record['geometry_passed']]
        if passed:
            selected = min(passed, key=lambda record: record['score'])
            save_json(out / 'candidate.json', selected)
            return selected
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=HERE / 'data/pickup_20261102_v1_run1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dart-xy', type=float, nargs=2)
    parser.add_argument('--dart-yaw', type=float, default=0., help='Initial world-Z rotation in degrees')
    parser.add_argument('--candidate-file', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--timestep', type=float, default=.0005)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = args.out / 'sources'
    sources.mkdir()
    hashes = {}
    for filename in ['retarget_pickup.py', 'try_pickup.py', 'retarget_arm.py', 'env.py']:
        path = HERE / filename
        shutil.copy2(path, sources / filename)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(args.out / 'provenance.json', dict(source_sha256=hashes, mujoco_version=mujoco.__version__,
              numpy_version=np.__version__, candidate_source_sha256=None if args.candidate_file is None else
              hashlib.sha256(args.candidate_file.read_bytes()).hexdigest()))
    video_time, closure = video_guidance(args.dataset, args.out)
    scene = PickupScene(args.timestep)
    if args.dart_xy is not None or args.dart_yaw:
        if args.dart_xy is not None:
            scene.config['dart_xy'] = args.dart_xy
        original = Rotation.from_quat(np.array(scene.config['dart_quat_wxyz'])[[1, 2, 3, 0]])
        rotated = Rotation.from_euler('z', args.dart_yaw, degrees=True) * original
        scene.config['dart_quat_wxyz'] = rotated.as_quat()[[3, 0, 1, 2]].tolist()
        scene.reset()
    bounds = np.array([geom_bounds(scene.model, scene.data, g) for g in scene.dart_geoms])
    center = np.array(scene.config['table_center'])[:2]
    half_size = np.array(scene.config['table_half_size'])[:2]
    if np.any(bounds[:, 0, :2].min(axis=0) < center - half_size + .005) or np.any(bounds[:, 1, :2].max(axis=0) > center + half_size - .005):
        raise ValueError('Initial dart footprint is outside the supported table area')
    save_json(args.out / 'reset.json', dict(dart_xy=scene.config['dart_xy'],
              dart_quat_wxyz=scene.config['dart_quat_wxyz'], settled_qpos=scene.data.qpos.tolist(),
              footprint_min=bounds[:, 0, :2].min(axis=0).tolist(), footprint_max=bounds[:, 1, :2].max(axis=0).tolist()))
    fitter = AcquisitionFit(scene)
    candidate = json.loads(args.candidate_file.read_text()) if args.candidate_file else fitter.fit_candidates(args.out)
    save_json(args.out / 'report.json', dict(geometry_passed=candidate is not None,
              physical_pickup_passed=False, model_sha256=scene.config['model_sha256'],
              stage='contact fitting', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if candidate is None or not args.run:
        return
    assessment = fitter.assess(np.array(candidate['q']))
    support = fitter.support()
    if not candidate['geometry_passed'] or not support['feasible'] or assessment['min_hand_table_height_m'] < .0002:
        raise ValueError('Candidate does not pass contact/clearance checks in this reset')
    squeeze = .005
    times, commands, planning = make_path(scene, candidate, squeeze=squeeze, lift_delta=(0., 0., .10))
    closed = np.r_[candidate['q'][7:], 1.55, 1.55]
    preshape = closed + np.array([0., .06, -.15, -.15, 0., 0.])
    observed_start, observed_end = np.interp([10.5, 13.6], video_time, closure)
    if observed_end - observed_start < .1:
        raise ValueError('Acquisition closure interval is unresolved')
    for index, time in enumerate(times):
        if 7 <= time <= 9:
            source_time = 10.5 + (time - 7) / 2 * (13.6 - 10.5)
            weight = np.clip((np.interp(source_time, video_time, closure) - observed_start) / (observed_end - observed_start), 0, 1)
            commands[index, 7:] = preshape + weight * (closed - preshape)
    # Check the actual modified reference, including intermediate coupled joints.
    minimum_clearance = np.inf
    violations = []
    controlled_joints = [scene.model.joint(name).id for name in ARM + FINGERS]
    low, high = scene.model.jnt_range[controlled_joints].T
    commands = np.clip(commands, low, high)
    for index, command in enumerate(commands):
        fitter.d.qpos[scene.aq] = command[:7]
        set_fingers(scene, fitter.d, command[7:])
        mujoco.mj_forward(scene.model, fitter.d)
        heights = [geom_z_bounds(scene.model, fitter.d, geom)[0] - scene.config['table_top_z'] for geom in fitter.hand_geoms]
        minimum_clearance = min(minimum_clearance, min(heights))
        for joint in range(scene.model.njnt):
            name = scene.model.joint(joint).name
            if scene.model.jnt_limited[joint] and (name.startswith('L_') or joint in scene.arm_ids):
                value = fitter.d.qpos[scene.model.jnt_qposadr[joint]]
                low, high = scene.model.jnt_range[joint]
                if value < low - 1e-7 or value > high + 1e-7:
                    violations.append([index, scene.model.joint(joint).name])
    planning.update(min_reference_hand_table_height_m=float(minimum_clearance),
                    joint_limit_violations=violations, video_closure_source_interval_s=[10.5, 13.6],
                    video_closure_robot_interval_s=[7., 9.])
    np.savez_compressed(args.out / 'commands.npz', time_s=times, arm_and_finger_q=commands)
    save_json(args.out / 'planning.json', planning)
    if minimum_clearance < 0 or violations or planning['max_wrist_ik_error_m'] > .01:
        save_json(args.out / 'report.json', dict(geometry_passed=True, physical_pickup_passed=False,
                  physics_executed=False, stage='reference rejected', planning=planning))
        raise ValueError('Reference fails clearance, limits, or wrist reachability')
    run_trial(scene, candidate, args.out.name, squeeze=squeeze, render=args.render, feedback=True, contact_trigger=True,
              command_path=(times, commands, planning), trial_directory=args.out,
              hold_force_targets=np.maximum(1.8 * np.array(support['required_normal_N']), [.8, .5, .5]),
              controller_label='Video closure guidance + contact-fitted robot motion')


if __name__ == '__main__':
    main()
