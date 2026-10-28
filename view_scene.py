"""Inspect the G1 darts room. Starts paused; 1-5 cameras, F orbit, Space physics, R reset."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from dart_model import DartDynamics

ROOT = Path(__file__).resolve().parent


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    data = mujoco.MjData(model)
    for name, angle in {"left_shoulder_roll_joint": .16, "right_shoulder_roll_joint": -.16,
                        "left_elbow_joint": .35, "right_elbow_joint": .65}.items():
        data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    # Place the lowest sole just on the lane surface in the inspection pose.
    soles = [g for g in range(model.ngeom) if "ankle_roll" in model.body(model.geom_bodyid[g]).name
             and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    if soles:
        bottom = min(data.geom_xpos[g, 2] - model.geom_size[g, 2] for g in soles)
        data.qpos[model.joint("floating_base_joint").qposadr[0]+2] += .009-bottom
        mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Save camera previews and exit")
    args = parser.parse_args()
    model, data = load_scene()
    dynamics = DartDynamics(model)
    if args.render:
        from PIL import Image
        output = ROOT / "previews"
        output.mkdir(exist_ok=True)
        with mujoco.Renderer(model, height=1000, width=1600) as renderer:
            for camera in ("room", "board", "robot", "side", "darts"):
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(output / f"{camera}.png")
        print(f"Previews: {output}")
        return
    initial = data.qpos.copy()
    state = {"paused": True, "reset": False, "camera": 0, "change_camera": True, "free": False}
    cameras = ["room", "board", "robot", "side", "darts"]
    def key(code):
        if code == 32:
            state["paused"] = not state["paused"]
            print("Paused" if state["paused"] else "Physics running (no balance policy)", flush=True)
        elif code in (82, 114):
            state["reset"] = True
        elif code in (49, 50, 51, 52, 53):
            state["camera"] = code-49
            state["change_camera"] = True
        elif code in (70, 102):
            state["free"] = True
    print("G1 Darts Lab | starts paused | 1 room, 2 board, 3 robot, 4 side, 5 darts | F orbit | Space physics | R reset", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            if state["reset"]:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = initial
                mujoco.mj_forward(model, data)
                state["paused"] = True
                state["reset"] = False
            if state["change_camera"]:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = model.camera(cameras[state["camera"]]).id
                state["change_camera"] = False
            if state["free"]:
                mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                viewer.cam.lookat[:] = [1.0, 0, 1.1]
                viewer.cam.distance = 5.3
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -19
                state["free"] = False
            if not state["paused"]:
                for _ in range(round((1/60)/model.opt.timestep)):
                    dynamics.step(data)
                if not np.isfinite(data.qpos).all():
                    state["reset"] = True
            viewer.sync()
            time.sleep(max(0, 1/60-(time.perf_counter()-started)))


if __name__ == "__main__":
    main()
