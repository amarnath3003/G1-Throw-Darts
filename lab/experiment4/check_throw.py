"""Replay the two-swing reference through contact physics and report release gates."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from prepare_reference import write_video
from retarget_arm import swing_checks
from inspire_grip import InspireGrip

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment2"))
from env import DartThrowEnv, impact_reward


def replay(reference, source, out, release_lead=.07):
    z = np.load(reference)
    t, q, frames = (z[key] for key in ("time", "q", "source_frame"))
    curve = PchipInterpolator(t, q, axis=0)
    derivative = curve.derivative()
    release_target = float(np.interp(225, frames, t))
    out.mkdir(parents=True, exist_ok=False)

    def trial(dt, held):
        sim = InspireGrip(timestep=dt, fixed_lower_body=True)
        m, d = sim.model, sim.data
        address = np.array([m.joint(str(name)).qposadr[0] for name in z["joints"]])
        dofs = np.array([m.joint(str(name)).dofadr[0] for name in z["joints"]])
        for name, value in zip(z["joints"], q[0]):
            sim.config["arm_pose"][str(name)] = float(value)
        sim.reset()
        if not held:
            d.qpos[sim.dart_q:sim.dart_q + 3] = [-1.5, 0., .15]
            sim.closed = False
            mujoco.mj_forward(m, d)
        pelvis = m.body("pelvis").id
        robot = set()
        for body in range(m.nbody):
            parent = body
            while parent:
                if parent == pelvis:
                    robot.add(body)
                    break
                parent = m.body_parentid[parent]
        scoring = SimpleNamespace(sim=sim, m=m, d=d, released=False, robot=robot,
                                  hand_bodies={i for i in robot if m.body(i).name.startswith("L_")},
                                  board_geom=m.geom("board_collision").id,
                                  tip_geom=m.geom("spare_dart_1_point").id)
        for _ in range(round(.25 / dt)):
            sim.step()
        initial_forces = sim.sample()["normal_N"]
        start = d.time
        command = None
        detached = None
        first_event = None
        clear_start = None
        release_speed = None
        bullseye = False
        rows = []
        wrist_y = []
        elbow_y = []
        elbow = m.body("left_elbow_link").id
        shoulder_y = float(d.xpos[m.body("left_shoulder_pitch_link").id, 1])
        head_geoms = [i for i in range(m.ngeom)
                      if m.geom_contype[i] and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                      and m.mesh(m.geom_dataid[i]).name == "head_link"]
        if len(head_geoms) != 1:
            raise ValueError("Expected one head collision mesh")
        def mesh_vertices(geom):
            mesh = m.geom_dataid[geom]
            start = m.mesh_vertadr[mesh]
            return m.mesh_vert[start:start + m.mesh_vertnum[mesh]]

        def world_bounds(geom, vertices):
            if vertices is None:
                # A bounding sphere encloses the dart's primitive barrel pieces.
                radius = m.geom_rbound[geom]
                return d.geom_xpos[geom] - radius, d.geom_xpos[geom] + radius
            points = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
            return points.min(axis=0), points.max(axis=0)

        head = head_geoms[0]
        head_low, head_high = world_bounds(head, mesh_vertices(head))
        dart_geoms = [(i, mesh_vertices(i) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH else None)
                      for i in np.flatnonzero(m.geom_bodyid == sim.dart)]

        def head_gap():
            # A positive box gap certifies separation, including for thin flight meshes.
            gaps = []
            for geom, vertices in dart_geoms:
                low, high = world_bounds(geom, vertices)
                gaps.append(np.max(np.maximum(low - head_high, head_low - high)))
            return float(min(gaps))

        head_clearance = head_gap() if held else None
        max_error = 0.
        maximum_penetration = 0.
        saturated = 0
        steps = round((t[-1] + 1.) / dt)
        snapshots = [d.qpos.copy()]
        next_frame = 1

        for index in range(steps):
            elapsed = d.time - start
            target_time = min(elapsed, t[-1])
            sim.target[address] = curve(target_time)
            if index == 0:
                sim.velocity_target = np.zeros(m.nv)
            sim.velocity_target[dofs] = derivative(target_time) if elapsed < t[-1] else 0.
            if held and command is None and elapsed >= release_target - release_lead:
                command = elapsed
                sim.closed = False
            sim.step()
            mujoco.mj_forward(m, d)
            elapsed = d.time - start
            max_error = max(max_error, float(np.max(abs(d.qpos[address] - sim.target[address]))))
            maximum_penetration = max([maximum_penetration] + [-c.dist for c in d.contact
                                      if all(m.geom_bodyid[g] in robot for g in c.geom)])
            saturated += bool(np.any(np.isclose(abs(d.ctrl[sim.ra]), m.actuator_ctrlrange[sim.ra, 1])))
            if held:
                head_clearance = min(head_clearance, head_gap())
                hand, hits = DartThrowEnv.contact_event(scoring)
                if hand:
                    clear_start = None
                elif clear_start is None:
                    clear_start = elapsed
                if clear_start is not None and elapsed - clear_start >= .01 - 1e-10 and detached is None:
                    detached = clear_start
                    scoring.released = True
                    jacobian = np.zeros((3, m.nv))
                    mujoco.mj_jacBodyCom(m, d, jacobian, None, sim.dart)
                    release_speed = float((jacobian @ d.qvel)[0])
                    if first_event is None and (command is None or detached < command):
                        first_event = "premature_slip"
                    if first_event is None and release_speed < .3:
                        first_event = "backward_or_drop"
                if hits and first_event is None:
                    first_event, point, tip_first = hits[0]
                    if first_event == "board":
                        goal = d.site_xpos[m.site("bullseye").id]
                        _, bullseye = impact_reward(np.linalg.norm((point - goal)[1:]), tip_first,
                                                   scoring.released and d.xmat[sim.dart, 0] > .5)
            if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
                first_event = first_event or "unstable"
                break
            rows.append((elapsed, d.xpos[sim.hand, 0], *d.qpos[address], *d.ctrl))
            wrist_y.append(d.xpos[sim.hand, 1])
            elbow_y.append(d.xpos[elbow, 1])
            if elapsed >= next_frame / 30 - dt / 2:
                snapshots.append(d.qpos.copy())
                next_frame += 1
        trace = np.asarray(rows)
        wrist = np.interp(t, np.r_[0., trace[:, 0]], np.r_[trace[0, 1], trace[:, 1]])
        checks = swing_checks(frames, wrist)
        swing_mask = trace[:, 0] <= t[-1]
        minimum_left = float(np.min(np.asarray(wrist_y)[swing_mask]))
        minimum_elbow = float(np.min(np.asarray(elbow_y)[swing_mask]))
        clean = bool(held and min(initial_forces) >= .02 and detached is not None and
                     command is not None and detached >= command and release_speed >= .3 and
                     first_event in (None, "board", "environment_hit") and head_clearance > 0.)
        label = f"{'held' if held else 'empty'}_{dt * 1000:g}ms"
        np.savez_compressed(out / f"{label}.npz", trace=trace, snapshots=snapshots)
        return sim, snapshots, {"trial": label, "shoulder_y_m": shoulder_y, "min_swing_wrist_y_m": minimum_left,
                                "min_swing_elbow_y_m": minimum_elbow,
                                "shoulder_lane_pass": min(minimum_left, minimum_elbow) >= shoulder_y,
                                "min_dart_head_box_gap_m": head_clearance, "swing_checks": checks, "initial_contact_N": initial_forces,
                                "command_s": command, "estimated_source_release_s": release_target,
                                "detached_s": detached, "forward_release_speed_m_s": release_speed,
                                "first_event": first_event, "clean_release": clean, "bullseye": bool(bullseye),
                                "max_joint_tracking_error_rad": max_error,
                                "max_robot_self_penetration_m": maximum_penetration,
                                "torque_saturated_fraction": saturated / max(len(trace), 1)}

    def render(sim, states, path, title):
        renderer = mujoco.Renderer(sim.model, 540, 720)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.75, .1, 1.22]
        camera.distance = 2.1
        camera.azimuth = -90
        camera.elevation = -6
        front = mujoco.MjvCamera()
        front.lookat[:] = [.3, .025, 1.07]
        front.distance = 1.5
        front.azimuth = 180
        front.elevation = 0

        def images():
            video = cv2.VideoCapture(str(source))
            image = np.zeros((540, 960, 3), dtype=np.uint8)
            try:
                for state in states:
                    ok, frame = video.read()
                    if ok:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sim.data.qpos[:] = state
                    mujoco.mj_forward(sim.model, sim.data)
                    renderer.update_scene(sim.data, camera)
                    side_image = renderer.render().copy()
                    renderer.update_scene(sim.data, front)
                    panels = [cv2.resize(image, (960, 540)), side_image, renderer.render()]
                    for panel, label in zip(panels, ("Source / GVHMR SMPL-X", title, "G1 front")):
                        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (15, 24, 34), -1)
                        cv2.putText(panel, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 2)
                    yield np.concatenate(panels, axis=1).tobytes()
            finally:
                video.release()

        try:
            write_video(path, (2400, 540), 30, images())
        finally:
            renderer.close()

    model = InspireGrip(fixed_lower_body=True)
    poses = np.tile(z["fixed_pose"], (len(q), 1))
    for i, name in enumerate(z["joints"]):
        poses[:, model.model.joint(str(name)).qposadr[0]] = q[:, i]
    # Hide the dart from the kinematic inspection by parking it off camera.
    poses[:, model.dart_q:model.dart_q + 3] = [-1.5, 0, .15]
    render(model, poses, out / "kinematic.mp4", "G1 kinematic reference")
    reports = []
    for held, dt in ((False, .0005), (True, .0005), (True, .00025)):
        sim, states, report = trial(dt, held)
        reports.append(report)
        print(json.dumps(report), flush=True)
        if held:
            render(sim, states, out / f"{report['trial']}.mp4", f"Physics | {report['trial']}")
    (out / "physics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "source", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--release-lead", type=float, default=.07)
    replay(**vars(parser.parse_args()))
