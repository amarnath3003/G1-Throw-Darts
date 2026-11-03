"""HaMeR-guided, bounded G1 arm IK and torque-driven clearance rehearsal.

This is an empty-hand arm test, not a successful pickup demonstration. HaMeR
supplies relative palm rotations; object-relative translations are designed in
the robot scene because monocular camera translations are not world calibrated.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from env import ARM, HERE, PickupScene, geom_z_bounds

OUT = HERE / 'research/arm_ik'


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*x*(10+x*(-15+6*x))


def geom_bounds(m, d, g):
    """World AABB, used only as a conservative separation lower bound."""
    r = d.geom_xmat[g].reshape(3, 3)
    p, size, kind = d.geom_xpos[g], m.geom_size[g], m.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[start:start+count] @ r.T + p
        return v.min(axis=0), v.max(axis=0)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(r) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.full(3, size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0]+size[1]*np.abs(r[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0]*np.linalg.norm(r[:, :2], axis=1)+size[1]*np.abs(r[:, 2])
    else:
        raise ValueError(f'Unsupported clearance geometry {kind}')
    return p-extent, p+extent


def palm_frame(j):
    """Proper frame: wrist -> middle MCP, index -> pinky across palm.

    Saved HaMeR joints already represent the left hand. Never mirror again.
    """
    x = j[9]-j[0]
    x /= np.linalg.norm(x)
    z = j[17]-j[5]
    z -= x*np.dot(x, z)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def hand_reference():
    path = HERE/'data/hamer_pilot/hand_raw.npz'
    raw = np.load(path)
    good = np.isfinite(raw['joints_wrist_relative']).all(axis=(1, 2))
    ts = raw['timestamps_s'][good]
    bases = np.array([palm_frame(j) for j in raw['joints_wrist_relative'][good]])
    # Anchor to approach, before occluded acquisition. These are relative
    # camera-frame changes, not a recovered camera-to-world transform.
    anchor = Rotation.from_matrix(bases[(ts >= 1) & (ts <= 2.25)]).mean().as_matrix()
    delta = Rotation.from_matrix(np.einsum('ij,njk->nik', anchor.T, bases)).as_rotvec()
    delta = gaussian_filter1d(delta, 1.2, axis=0)
    magnitude = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, np.deg2rad(65)/np.maximum(magnitude, 1e-9))[:, None]
    # Reduce influence of uncertain monocular orientation, especially during
    # occlusion. All raw outputs are retained separately.
    confidence = np.where((ts >= 2.25) & (ts <= 4.75), .20, .45)
    delta *= confidence[:, None]
    return ts, Slerp(ts, Rotation.from_rotvec(delta)), dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), valid_frames=int(good.sum()),
        side='left, already reflected by inference; no additional reflection',
        camera_motion_recovered=False, translation='designed object-relative path',
        orientation='relative palm-frame changes; smoothed, capped 65 deg then weighted 0.20/0.45',
        anatomical_to_robot_basis='diag(1,-1,-1): wrist-to-middle +X, index-to-pinky -Z; proper rotation, not reflection',
        finger_motion='existing open preshape; finger retargeting/contact acquisition not implemented')


class ArmIK:
    def __init__(self, scene):
        self.s = s = scene
        self.m = m = s.model
        self.d = d = mujoco.MjData(m)
        d.qpos[:] = s.data.qpos
        self.lo, self.hi = m.jnt_range[s.arm_ids].T.copy()
        self.margin = .10
        self.lo += self.margin
        self.hi -= self.margin
        self.elbow = m.body('left_elbow_link').id
        self.shoulder = m.body('left_shoulder_pitch_link').id
        self.preferred = np.array([.20, .25, .0, -.55, 1.30, .0, .0])
        self.weights = np.array([.025, .035, .035, .035, .065, .12, .12])
        self.arm_bodies = {m.body(n.replace('_joint', '_link')).id for n in ARM}
        self.moving_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and
                             (m.geom_bodyid[g] in self.arm_bodies or m.body(m.geom_bodyid[g]).name.startswith('L_'))]
        tables = [g for g in range(m.ngeom) if m.geom(g).name.startswith('table_')]
        torso = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name == 'torso_link']
        distal = [g for g in self.moving_geoms if m.body(m.geom_bodyid[g]).name.startswith(('L_', 'left_wrist', 'left_elbow'))]
        self.pairs = [(g, t) for g in self.moving_geoms for t in tables] + [(g, t) for g in distal for t in torso]
        self.table_pairs = [(g, m.geom('table_top').id) for g in self.moving_geoms]
        mujoco.mj_forward(m, d)
        self.elbow_target = d.xpos[self.shoulder]+np.array([.005, .10, -.17])
        # A point between the two long fingers, in wrist-link coordinates.
        r = d.xmat[s.hand].reshape(3, 3)
        tip = np.mean([d.xpos[m.body(n).id] for n in ('L_index_intermediate', 'L_middle_intermediate')], axis=0)
        self.tool = r.T@(tip-d.xpos[s.hand])

    def forward(self, q):
        self.d.qpos[self.s.aq] = q
        mujoco.mj_forward(self.m, self.d)
        r = self.d.xmat[self.s.hand].reshape(3, 3)
        return self.d.xpos[self.s.hand]+r@self.tool, r

    def distances(self, pairs=None, data=None):
        d = self.d if data is None else data
        pairs = self.pairs if pairs is None else pairs
        bounds = {g: geom_bounds(self.m, d, g) for g in {g for pair in pairs for g in pair}}
        result = []
        for a, b in pairs:
            native = mujoco.mj_geomDistance(self.m, d, a, b, 1., None)
            # AABB separation is a lower bound on Euclidean distance.
            # MuJoCo 3.10 occasionally returns zero for separated mesh/box pairs.
            gap = np.linalg.norm(np.maximum(np.maximum(bounds[a][0]-bounds[b][1], bounds[b][0]-bounds[a][1]), 0))
            result.append(max(native, gap) if gap > 0 else native)
        return np.asarray(result)

    def solve(self, point, rot, previous, multistart=False):
        def residual(q):
            p, r = self.forward(q)
            # Exact mesh/primitive vertical support is stable under numerical
            # differentiation. Native mesh distance can be discontinuous at its
            # search cutoff; use the tabletop plane conservatively in the fit,
            # then validate against all finite table and torso geoms separately.
            heights = np.array([geom_z_bounds(self.m, self.d, g)[0] for g in self.moving_geoms])
            collision = np.maximum(self.s.config['table_top_z']+.018-heights, 0)*60
            return np.r_[35*(p-point), 1.2*Rotation.from_matrix(rot.T@r).as_rotvec(),
                         self.weights*(q-self.preferred), .05*(q-previous),
                         .8*(self.d.xpos[self.elbow]-self.elbow_target), collision]
        seeds = [previous]
        if multistart:
            seeds += [self.s.data.qpos[self.s.aq], np.array([.5, .3, -.5, -.7, 1.3, 0, 0]),
                      np.array([.5, .4, .5, -.7, 1.3, 0, 0])]
        fits = [least_squares(residual, np.clip(q, self.lo+1e-6, self.hi-1e-6),
                              bounds=(self.lo, self.hi), max_nfev=200, ftol=1e-7, diff_step=.002, jac='3-point') for q in seeds]
        fit = min(fits, key=lambda f: np.dot(f.fun, f.fun))
        p, r = self.forward(fit.x)
        return fit.x, dict(position_error_m=float(np.linalg.norm(p-point)),
                          orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(rot.T@r).magnitude())),
                          min_environment_distance_m=float(self.distances().min()),
                          elbow_position=self.d.xpos[self.elbow].tolist())


def build(s):
    ik = ArmIK(s)
    ts, orient, provenance = hand_reference()
    source_times = np.linspace(1, 7.135, 93)
    dart = s.data.xpos[s.dart].copy()
    # Palm down, fingers forward and slightly downward. Compared with the old
    # vertical throwing-grip alignment this allows a low, bent elbow.
    base = Rotation.from_euler('y', 18, degrees=True).as_matrix() @ Rotation.from_euler('x', 90, degrees=True).as_matrix()
    qs, points, rotations, diagnostics = [], [], [], []
    previous = ik.preferred.copy()
    for i, t in enumerate(source_times):
        reach = smoothstep((t-1)/2.0)
        lift = smoothstep((t-4.75)/1.7)
        point = dart + np.array([-.020*(1-reach)+.060*lift, -.015*(1-reach), .055+.080*(1-reach)+.080*lift])
        basis = np.diag([1., -1., -1.])
        rot = base @ basis @ orient(np.clip(t, ts[0], ts[-1])).as_matrix() @ basis.T
        q, diag = ik.solve(point, rot, previous, multistart=(i == 0))
        qs.append(q); points.append(point); rotations.append(rot); diagnostics.append(diag)
        previous = q
        if i % 20 == 0:
            print(f'IK {i+1}/{len(source_times)} error={diag["position_error_m"]*1000:.1f} mm clearance={diag["min_environment_distance_m"]*1000:.1f} mm', flush=True)
    # Start at the existing physical reset; use a smooth joint-space transition.
    # A densely sampled collision gate checks this segment before execution.
    lead_t = np.linspace(0, 4, 81, endpoint=False)
    lead = s.data.qpos[s.aq]+smoothstep(lead_t/4)[:, None]*(qs[0]-s.data.qpos[s.aq])
    times = np.r_[lead_t, 4+(source_times-1)*2.0]
    q = np.vstack([lead, qs])
    np.savez_compressed(OUT/'trajectory.npz', time_s=times, arm_q=q, source_time_s=source_times,
                        target_point=np.array(points), target_rotation=np.array(rotations), tool_point_local=ik.tool)
    report = dict(scope='empty-hand arm reach/hover/lift rehearsal; no grasp or object lift',
                  source=provenance, joint_names=ARM, model_sha256=s.config['model_sha256'],
                  joint_limit_margin_rad=.10, reference_diagnostics=diagnostics,
                  duration_s=float(times[-1]), planned_max_position_error_m=max(x['position_error_m'] for x in diagnostics),
                  planned_max_orientation_error_deg=max(x['orientation_error_deg'] for x in diagnostics))
    clearance, limit = [], []
    for t in np.arange(0, times[-1]+.0001, .02):
        cmd = np.array([np.interp(t, times, q[:, j]) for j in range(7)])
        ik.forward(cmd)
        clearance.append(float(ik.distances().min()))
        limit.append(float(np.min(np.r_[cmd-ik.lo, ik.hi-cmd])))
    report['dense_kinematic_min_clearance_m'] = min(clearance)
    report['joint_speed_max_rad_s'] = float(np.max(np.abs(np.diff(q, axis=0)/np.diff(times)[:, None])))
    report['joint_limit_margin_actual_rad'] = min(limit)+ik.margin
    report['planning_pass'] = bool(min(clearance) >= .005 and min(limit) >= -1e-8 and
                                   report['joint_speed_max_rad_s'] < 1.5 and report['planned_max_position_error_m'] < .015 and
                                   report['planned_max_orientation_error_deg'] < 20)
    return times, q, ik, report


def rollout(s, times, q, ik, render=False):
    """Only actuator targets change during execution; qpos is never assigned."""
    sys.path.insert(0, str(HERE.parent))
    from dart_model import DartDynamics, properties, load_profile
    for filename in ('dart_model.py', 'dart_profile.json'):
        if hashlib.sha256((HERE.parent/filename).read_bytes()).hexdigest() != s.config['source_sha256'][filename]:
            raise RuntimeError(f'{filename} changed since pinned scene; review before transport')
    m, d = s.model, s.data
    # mj_saveLastXML rounded inertia to six significant digits in the pinned
    # snapshot. Restore only this sub-5-ppm serialization error in memory so the
    # pinned aerodynamic model's strict property check can be honored.
    expected = np.asarray(properties(load_profile())['inertia_kg_m2'])
    original = m.body_inertia[s.dart].copy()
    if not np.allclose(original, expected, rtol=5e-6, atol=0):
        raise RuntimeError('Dart inertia discrepancy exceeds XML-rounding allowance')
    if not np.allclose(m.body_iquat[s.dart], [1, 0, 0, 0], atol=1e-10):
        raise RuntimeError('Dart principal inertia frame changed')
    m.body_inertia[s.dart] = expected
    mujoco.mj_setConst(m, mujoco.MjData(m))
    mujoco.mj_forward(m, d)
    air = DartDynamics(m)
    start = d.time
    dart_start = d.xpos[s.dart].copy()
    velocities, errors, clearances, penetrations, tool_errors = [], [], [], [], []
    penetration_count = 0
    samples, states, actual_times = [], [], []
    next_sample = 0.
    baseline = {tuple(sorted((int(c.geom1), int(c.geom2)))) for c in d.contact if c.dist < -.0002}
    while d.time-start < times[-1]+1:
        t = d.time-start
        cmd = np.array([np.interp(t, times, q[:, j]) for j in range(7)])
        s.target[s.aq] = cmd
        air.apply(d)
        s.step()
        errors.append(float(np.max(np.abs(d.qpos[s.aq]-cmd))))
        velocities.append(float(np.max(np.abs(d.qvel[m.jnt_dofadr[s.arm_ids]]))))
        # Inspect contact output at every physics step, not only video frames.
        for c in d.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            if c.dist < -.0002 and pair not in baseline and any(g in ik.moving_geoms for g in pair):
                penetration_count += 1
                if len(penetrations) < 20:
                    penetrations.append(dict(time_s=float(t), geom_ids=list(pair), distance_m=float(c.dist)))
        if t+1e-9 >= next_sample:
            mujoco.mj_forward(m, d)
            clearance = float(ik.distances(data=d).min())
            clearances.append(clearance)
            planned_point, _ = ik.forward(cmd)
            actual_point = d.xpos[s.hand]+d.xmat[s.hand].reshape(3, 3)@ik.tool
            tool_errors.append(float(np.linalg.norm(planned_point-actual_point)))
            samples.append(d.xpos[s.hand].copy()); states.append(d.qpos.copy()); actual_times.append(t)
            next_sample += 1/30
    result = dict(timestep_s=float(m.opt.timestep), max_joint_tracking_error_rad=max(errors),
                  dart_inertia_rounding_restored_relative=float(np.max(np.abs(original/expected-1))),
                  max_joint_speed_rad_s=max(velocities), min_environment_clearance_m=min(clearances),
                  max_tool_tracking_error_m=max(tool_errors), clearance_sampling_hz=30,
                  contact_check='every physics step, new arm penetration threshold 0.2 mm',
                  new_arm_penetration_count=penetration_count, penetrating_contacts=penetrations,
                  dart_displacement_m=float(np.linalg.norm(d.xpos[s.dart]-dart_start)),
                  pickup_success=False, finite_state=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()))
    result['pass'] = bool(result['finite_state'] and min(clearances) > .005 and not penetrations and
                          max(errors) < .10 and result['dart_displacement_m'] < .002)
    np.savez_compressed(OUT/f'rollout_{m.opt.timestep:g}.npz', time_s=actual_times, qpos=states, wrist_position=samples)
    if render:
        render_rollout(s, np.asarray(states), np.asarray(actual_times))
    return result


def render_rollout(s, states, times):
    import cv2
    m = s.model
    rd = mujoco.MjData(m)
    cameras = []
    for azimuth in (135, 270):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.40, .17, 1.00]
        cam.distance = 1.65; cam.azimuth = azimuth; cam.elevation = -18
        if azimuth == 270:
            cam.lookat[:] = [.40, .18, .97]
            cam.distance = 1.20
        cameras.append(cam)
    target = OUT/'g1_arm_ik_rehearsal.mp4'
    proc = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', '1280x480', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
                             '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE)
    with mujoco.Renderer(m, height=480, width=640) as renderer:
        for i, (state, t) in enumerate(zip(states, times)):
            rd.qpos[:] = state
            mujoco.mj_forward(m, rd)
            panels = []
            for cam in cameras:
                renderer.update_scene(rd, camera=cam)
                panels.append(renderer.render().copy())
            frame = np.hstack(panels)
            label = 'Settle into natural arm posture' if t < 4 else ('Reach / hover over dart' if t < 11.5 else 'Hand lift / wrist rotation rehearsal')
            cv2.rectangle(frame, (0, 0), (1280, 54), (22, 25, 32), -1)
            cv2.putText(frame, f'G1 + HaMeR-guided IK | {label} | {t:.1f}s', (18, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 240, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Torque-driven simulation | Empty hand: grasp/contact acquisition still pending', (18, 44), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 205, 120), 1, cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
            if i in (0, int(6*30), int(8*30), len(states)-1):
                cv2.imwrite(str(OUT/f'preview_{i:04d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--validate-timesteps', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = PickupScene()
    times, q, ik, report = build(s)
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['planning_pass']:
        raise SystemExit('Planning gate failed; inspect report before execution')
    report['physics'] = [rollout(s, times, q, ik, args.render)]
    if args.validate_timesteps:
        finer = PickupScene(.00025)
        report['physics'].append(rollout(finer, times, q, ArmIK(finer)))
    report['pass'] = all(x['pass'] for x in report['physics'])
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'reference_diagnostics'}, indent=2))
    if not report['pass']:
        raise SystemExit('Physics validation failed')


if __name__ == '__main__':
    main()
