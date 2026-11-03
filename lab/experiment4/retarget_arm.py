"""Fit the reconstructed arm to the supported Inspire G1 and audit the two draw-backs."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment1_throw import ARM
from inspire_grip import InspireGrip


def swing_checks(source_frames, forward):
    checks = []
    for begin, back, reset in (((103, 119), (137, 154), (165, 179)),
                               ((165, 179), (198, 216), (229, 247))):
        def values(window):
            return forward[(source_frames >= window[0]) & (source_frames <= window[1])]
        start, rear, finish = map(values, (begin, back, reset))
        if not all(len(part) for part in (start, rear, finish)):
            checks.append({"passed": False, "reason": "incomplete swing"})
            continue
        retreat = float(np.median(start) - np.median(rear))
        advance = float(np.median(finish) - np.median(rear))
        checks.append({"retreat_m": retreat, "advance_m": advance,
                       "passed": retreat >= .03 and advance >= .03})
    return checks


def fit(human_path, out, left_heading_deg=12.):
    human = np.load(human_path)
    sim = InspireGrip(fixed_lower_body=True)
    m, d = sim.model, sim.data
    joints = np.array([m.joint(name).id for name in ARM])
    addresses = m.jnt_qposadr[joints]
    elbow, wrist = [m.body(name).id for name in ("left_elbow_link", "left_wrist_yaw_link")]
    shoulder = m.body("left_shoulder_pitch_link").id
    fixed = d.qpos.copy()
    upper_length = np.linalg.norm(d.xpos[elbow] - d.xpos[shoulder])
    lower_length = np.linalg.norm(d.xpos[wrist] - d.xpos[elbow])
    origin = d.xpos[shoulder].copy()
    # Keep one heading for the whole clip, with a leftward bias around the shoulder.
    human_forward = np.median((human["upper"] + human["lower"])[-8:, :2], axis=0)
    heading = np.arctan2(human_forward[1], human_forward[0])
    alignment = Rotation.from_euler("z", np.deg2rad(left_heading_deg) - heading).as_matrix()
    upper = human["upper"] @ alignment.T
    lower = human["lower"] @ alignment.T
    target_elbow = origin + upper * upper_length
    target_wrist = target_elbow + lower * lower_length
    # Measure the front-view lane from the actual shoulder, not the pelvis.
    shoulder_y = origin[1]
    target_wrist[:, 1] = np.maximum(shoulder_y + .045, target_wrist[:, 1])
    target_elbow[:, 1] = np.maximum(shoulder_y + .025, target_elbow[:, 1])
    low, high = m.jnt_range[joints].T
    low, high = low + .015, high - .015
    last = np.clip(fixed[addresses], low, high)
    hand_initial = d.xmat[wrist].reshape(3, 3).copy()
    orientation = human["wrist_rotation"]
    rotations = alignment @ orientation @ orientation[0].T @ alignment.T @ hand_initial
    q, measured, elbows, errors, penetration = [], [], [], [], []
    robot = set()
    pelvis = m.body("pelvis").id
    for body in range(m.nbody):
        parent = body
        while parent:
            if parent == pelvis:
                robot.add(body)
                break
            parent = m.body_parentid[parent]

    def self_penetration():
        return max([0.] + [-c.dist for c in d.contact
                          if all(m.geom_bodyid[g] in robot for g in c.geom)])

    # Add clearance in the IK contact query only. Physics replay uses the original model.
    clearance = .025
    dart_geoms = m.geom_bodyid == sim.dart
    m.geom_margin[dart_geoms] = np.maximum(m.geom_margin[dart_geoms], clearance)
    grip_pos = np.asarray(sim.config["dart_pos_hand"])
    grip_quat = np.asarray(sim.config["dart_quat_hand"])
    dart_quat = np.zeros(4)
    dart_ends = [m.geom(name).id for name in ("spare_dart_1_point", "spare_dart_1_flight_0")]
    for index, (frame, e, w, rotation) in enumerate(
        zip(human["source_frame"], target_elbow, target_wrist, rotations)
    ):
        def error(candidate):
            d.qpos[addresses] = candidate
            mujoco.mj_forward(m, d)
            dart_penetration = 0.
            dart_lane = np.zeros(2)
            dart_direction = np.zeros(3)
            if frame <= 226:
                hand_rotation = d.xmat[wrist].reshape(3, 3)
                d.qpos[sim.dart_q:sim.dart_q + 3] = d.xpos[wrist] + hand_rotation @ grip_pos
                mujoco.mju_mulQuat(dart_quat, d.xquat[wrist], grip_quat)
                d.qpos[sim.dart_q + 3:sim.dart_q + 7] = dart_quat
                mujoco.mj_forward(m, d)
                dart_lane = np.minimum(d.geom_xpos[dart_ends, 1] - shoulder_y - .025, 0.)
                dart_direction = d.xmat[sim.dart].reshape(3, 3)[:, 0] - [.98, 0., .20]
                for contact in d.contact:
                    a, b = contact.geom
                    if m.geom_bodyid[b] == sim.dart:
                        a, b = b, a
                    if m.geom_bodyid[a] != sim.dart:
                        continue
                    if m.geom_bodyid[b] in sim.distal and "_barrel_" in m.geom(a).name:
                        continue
                    dart_penetration = max(dart_penetration, clearance - contact.dist)
            angle = Rotation.from_matrix(rotation.T @ d.xmat[wrist].reshape(3, 3)).as_rotvec()
            return np.r_[d.xpos[elbow] - e, 1.5 * (d.xpos[wrist] - w),
                         3 * min(d.xpos[elbow, 1] - shoulder_y, 0.),
                         4 * min(d.xpos[wrist, 1] - shoulder_y - .025, 0.),
                         .012 * angle, .013 * (candidate - last), 3 * dart_lane,
                         .035 * dart_direction, 24 * self_penetration(), 24 * dart_penetration]
        # Limit frame-to-frame jumps before interpolation and torque replay.
        delta = .22
        bounds = (np.maximum(low, last - delta), np.minimum(high, last + delta)) if index else (low, high)
        solution = least_squares(error, last, bounds=bounds, max_nfev=80, diff_step=1e-4)
        last = solution.x
        error(last)
        q.append(last.copy())
        measured.append(d.xpos[wrist].copy())
        elbows.append(d.xpos[elbow].copy())
        errors.append(np.linalg.norm(d.xpos[wrist] - w))
        penetration.append(max([0.] + [-c.dist for c in d.contact
                          if all(m.geom_bodyid[g] in robot for g in c.geom)]))

    time = human["time"]
    curve = PchipInterpolator(time, q, axis=0)
    velocity = curve.derivative()(time)
    checks = swing_checks(human["source_frame"], np.asarray(measured)[:, 0])
    human_checks = swing_checks(human["source_frame"], target_wrist[:, 0])
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "reference.npz", time=time, q=q, velocity=velocity,
                        joints=ARM, fixed_pose=fixed, source_frame=human["source_frame"],
                        wrist=measured, elbow=elbows, wrist_target=target_wrist)
    report = {"left_heading_deg": left_heading_deg,
              "shoulder_y_m": float(shoulder_y),
              "minimum_elbow_y_m": float(np.min(np.asarray(elbows)[:, 1])),
              "minimum_wrist_y_m": float(np.min(np.asarray(measured)[:, 1])),
              "pullback_wrist_y_m": [float(np.median(np.asarray(measured)[
                  (human["source_frame"] >= begin) & (human["source_frame"] <= end), 1]))
                  for begin, end in ((137, 154), (198, 216))],
              "human_swing_checks": human_checks, "robot_swing_checks": checks,
              "wrist_error_median_m": float(np.median(errors)), "wrist_error_max_m": float(max(errors)),
              "max_self_penetration_m": float(max(penetration)),
              "peak_reference_speed_rad_s": np.max(np.abs(curve.derivative()(
                  np.linspace(time[0], time[-1], 1000))), axis=0).tolist()}
    (out / "retarget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("human_path", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--left-heading-deg", type=float, default=12.)
    fit(**vars(parser.parse_args()))
