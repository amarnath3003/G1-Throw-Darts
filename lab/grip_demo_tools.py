"""Validation, rendering and interactive joint editor for grip_demo.py."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import time

import mujoco
import numpy as np

from grip_demo import ARM_NAMES, GripDemo, OUT, POSE, ROOT


def advance(sim, seconds, rows=None):
    for i in range(round(seconds/sim.model.opt.timestep)):
        sim.step()
        if rows is not None and i % round(.01/sim.model.opt.timestep) == 0:
            rows.append(sim.sample())
    assert np.isfinite(sim.data.qpos).all(), "Non-finite physics state"
    assert not any(w.number for w in sim.data.warning), "MuJoCo emitted a physics warning"


def validate():
    OUT.mkdir(exist_ok=True)
    sim = GripDemo()
    m, d = sim.model, sim.data
    assert m.jnt_type[m.joint("spare_dart_1_free").id] == mujoco.mjtJoint.mjJNT_FREE
    assert m.neq == 0, "Demo must not contain attachment constraints"
    rows = []
    advance(sim, 10, rows)
    hold = sim.sample()
    weight = m.body_mass[sim.dart]*9.81
    settled = [r for r in rows if r["time_s"] > 1.]
    slip = max(abs(r["com_z_m"]-sim.initial_z) for r in settled)
    assert slip < .005, f"Dart slipped {slip:.6f} m"
    assert min(r["thumb_normal_N"] for r in settled) > .5
    assert min(r["index_normal_N"] for r in settled) > .5
    assert all(not r["other_dart_contacts"] for r in settled), "An unintended surface supported the dart"
    assert abs(hold["contact_vertical_N"]-weight) < .01
    assert hold["aim_error_deg"] < 1.
    assert hold["tip_down_cosine"] > .995
    release_time = d.time
    release_z = hold["com_z_m"]
    sim.closed = False
    advance(sim, .3, rows)
    fall = sim.sample()
    assert fall["thumb_normal_N"]+fall["index_normal_N"] == 0.
    assert release_z-fall["com_z_m"] > .35
    assert fall["vz_m_s"] < -2.5
    # Check measured free-flight acceleration after contact clears, before impact.
    free = [r for r in rows if release_time+.08 < r["time_s"] < release_time+.25]
    acceleration = np.polyfit([r["time_s"] for r in free], [r["vz_m_s"] for r in free], 1)[0]
    assert -10.1 < acceleration < -9.4, acceleration
    advance(sim, 1., rows)
    landed = sim.sample()
    assert landed["com_z_m"] < .13
    assert "drop_mat" in landed["other_dart_contacts"]
    report = {"passed": True, "mujoco_version": mujoco.__version__,
              "dart_mass_kg": float(m.body_mass[sim.dart]), "weight_N": float(weight),
              "hold_duration_s": 10, "maximum_hold_height_change_m": slip,
              "hold": hold, "after_0_3s_release": fall,
              "free_flight_acceleration_m_s2": float(acceleration), "landed": landed,
              "attachment_constraints": int(m.neq), "pelvis_support": "fixed physical test fixture",
              "gripper": "custom two slide-joint fingers; not a calibrated commercial hand",
              "negative_controls": {}, "robustness": []}
    with (OUT / "trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for mode in ("no_squeeze", "frictionless", "initially_open"):
        test = GripDemo()
        if mode == "no_squeeze":
            test.force = 0.
        elif mode == "frictionless":
            ids = [test.model.geom(f"pinch_{s}_pad").id for s in ("thumb", "index")]
            test.model.geom_friction[ids] = 0.
        else:
            test.reset(closed=False)
        advance(test, .3)
        result = test.sample()
        drop = test.initial_z-result["com_z_m"]
        # Without squeeze/friction the barrel must slip. The wider flight can
        # subsequently catch on the still-closed physical jaws; that is expected.
        minimum_drop = .35 if mode == "initially_open" else .03
        assert drop > minimum_drop, f"{mode}: barrel failed to slip as expected"
        report["negative_controls"][mode] = {"drop_in_0_3s_m": drop, "state": result,
            "criterion": "free fall" if mode == "initially_open" else "barrel loses its grip; flight may catch on closed jaws"}
    for dt, force in ((.00025, .8), (.0005, .4), (.0005, 1.2)):
        test = GripDemo(dt)
        test.force = force
        advance(test, 2.)
        held = test.sample()
        assert abs(test.initial_z-held["com_z_m"]) < .005
        test.closed = False
        advance(test, .3)
        released = test.sample()
        assert held["com_z_m"]-released["com_z_m"] > .35
        report["robustness"].append({"timestep_s": dt, "command_per_finger_N": force,
                                     "hold": held, "released": released})
    (OUT / "validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (OUT / "pose.json").write_text(json.dumps({"source": "manually posed, wrist alignment refined geometrically",
        "joints_rad": POSE, "joint_degrees": {k:math.degrees(v) for k,v in POSE.items()},
        "video_used": False}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def camera(sim, name):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if name == "grip":
        cam.lookat[:] = sim.data.site_xpos[sim.site] + [0, 0, .005]
        cam.distance, cam.azimuth, cam.elevation = .36, 125, -16
    elif name == "drop":
        cam.lookat[:] = [.85, -.08, .70]
        cam.distance, cam.azimuth, cam.elevation = 1.65, 110, -10
    elif name == "robot":
        cam.lookat[:] = [.62, -.02, .84]
        cam.distance, cam.azimuth, cam.elevation = 2.15, 120, -12
    else:
        cam.lookat[:] = [1.4, 0, 1.05]
        cam.distance, cam.azimuth, cam.elevation = 4.8, 130, -16
    return cam


def render():
    from PIL import Image, ImageDraw, ImageFont
    OUT.mkdir(exist_ok=True)
    sim = GripDemo()
    advance(sim, .5)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 23)
        small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
    except OSError:
        font, small = ImageFont.load_default(), ImageFont.load_default()
    width, height = 1440, 810
    video = OUT / "grip_release.mp4"
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "30", "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    captures = {}
    try:
        with mujoco.Renderer(sim.model, height=630, width=720) as renderer:
            for frame in range(180):
                t = frame/30
                if frame:
                    # 30 fps output, with simulation stepped at its own exact timestep.
                    target = .5 + t
                    while sim.data.time < target-1e-9:
                        sim.step()
                if frame == 90:
                    sim.closed = False
                state = sim.sample()
                canvas = Image.new("RGB", (width, height), "#111c25")
                draw = ImageDraw.Draw(canvas)
                title = "HOLD  /  contact force" if sim.closed else "RELEASE  /  gravity"
                draw.text((30, 16), "G1  |  PINCH & DROP", fill="#f1f4ef", font=font)
                draw.text((880, 16), title, fill="#72ded0" if sim.closed else "#eebc74", font=font)
                for x, view in ((0, "robot"), (720, "grip" if sim.closed else "drop")):
                    renderer.update_scene(sim.data, camera=camera(sim, view))
                    canvas.paste(Image.fromarray(renderer.render()), (x, 66))
                draw.text((30, 716), f"t = {t:4.2f} s     Dart 26.7 g     Tip down     Base supported", font=small, fill="#d9e3e8")
                draw.text((30, 753), f"Pad normal forces: {state['thumb_normal_N']:.3f} N  /  {state['index_normal_N']:.3f} N", font=small, fill="#72ded0")
                draw.text((760, 716), f"Height {state['com_z_m']:.3f} m    Vertical speed {state['vz_m_s']:.3f} m/s", font=small, fill="#d9e3e8")
                draw.text((760, 753), "Custom 2-finger test gripper | MuJoCo contact + friction", font=small, fill="#9fb2c0")
                proc.stdin.write(np.asarray(canvas).tobytes())
                if frame in (60, 99, 140):
                    name = {60:"holding", 99:"falling", 140:"landed"}[frame]
                    canvas.save(OUT / f"{name}.png")
                    captures[name] = state
                if frame % 30 == 0:
                    print(f"Rendered {frame}/180 frames", flush=True)
    finally:
        proc.stdin.close()
        result = proc.wait()
    if result:
        raise RuntimeError(f"ffmpeg exited {result}")
    (OUT / "render_states.json").write_text(json.dumps(captures, indent=2)+"\n", encoding="utf-8")
    print(video, flush=True)


def viewer():
    import tkinter as tk
    from tkinter import ttk
    import mujoco.viewer
    sim = GripDemo()
    window = tk.Tk()
    window.title("G1 grip controls | joints, force, release")
    window.geometry("490x740+30+65")
    window.configure(bg="#15222c")
    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure("TFrame", background="#15222c")
    style.configure("TLabel", background="#15222c", foreground="#e6ecef", font=("Segoe UI", 10))
    frame = ttk.Frame(window, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="CONTACT GRIP / GRAVITY RELEASE", font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Custom two-finger gripper  |  Supported G1 base").pack(anchor="w", pady=(4, 12))
    state = {"paused": False, "reset": False, "default": False, "camera": "grip", "dirty_camera": True,
             "action": None, "cycle_release": None, "alive": True}
    def command(action):
        state["action"] = action
    row = ttk.Frame(frame)
    row.pack(fill="x")
    for label, action in (("C  Close", "close"), ("O  Open / drop", "open"), ("R  Reload dart", "reset")):
        ttk.Button(row, text=label, command=lambda a=action: command(a)).pack(side="left", padx=3)
    ttk.Button(frame, text="Run demonstration: hold 3 s, then release", command=lambda: command("cycle")).pack(fill="x", pady=8)
    ttk.Label(frame, text="Reload places a new dart. Closing does not retrieve it.").pack(anchor="w")
    force = tk.DoubleVar(value=sim.force)
    force_label = ttk.Label(frame, text="Grip force: 0.80 N per finger")
    force_label.pack(anchor="w", pady=(12, 0))
    def change_force(value):
        sim.force = float(value)
        force_label.configure(text=f"Grip force: {sim.force:.2f} N per finger")
    ttk.Scale(frame, from_=0, to=2., variable=force, command=change_force).pack(fill="x")
    ttk.Label(frame, text="Arm joint targets (degrees) — powered servos").pack(anchor="w", pady=(12, 5))
    variables = {}
    for name in ARM_NAMES:
        line = ttk.Frame(frame)
        line.pack(fill="x", pady=2)
        label = name.replace("right_", "").replace("_joint", "").replace("_", " ")
        ttk.Label(line, text=label, width=16).pack(side="left")
        var = tk.DoubleVar(value=math.degrees(sim.pose[name]))
        variables[name] = var
        number = ttk.Label(line, text=f"{var.get():.1f}", width=6)
        number.pack(side="right")
        def adjust(value, n=name, label=number):
            sim.set_joint(n, math.radians(float(value)))
            label.configure(text=f"{float(value):.1f}")
        lo, hi = np.degrees(sim.model.jnt_range[sim.model.joint(name).id])
        ttk.Scale(line, from_=float(lo), to=float(hi), variable=var, command=adjust).pack(fill="x", expand=True)
    ttk.Button(frame, text="Restore bullseye pose + reload dart", command=lambda: command("default")).pack(fill="x", pady=10)
    views = ttk.Frame(frame)
    views.pack(fill="x")
    def set_view(name):
        state.update(camera=name, dirty_camera=True)
    for label, name in (("1 Grip", "grip"), ("2 Drop", "drop"), ("3 Robot", "robot"), ("4 Room", "room")):
        ttk.Button(views, text=label, command=lambda n=name: set_view(n)).pack(side="left", padx=2)
    ttk.Button(frame, text="Space  Pause / resume", command=lambda: command("pause")).pack(fill="x", pady=8)
    status = ttk.Label(frame, text="Starting physics...", justify="left")
    status.pack(anchor="w", pady=8)
    ttk.Label(frame, text="The dart is free: only contact, friction, gravity and air loads.\nJoint changes may lose the grip. Reload uses the current pose.",
              font=("Segoe UI", 9)).pack(anchor="w", pady=5)
    def close_window():
        state["alive"] = False
    window.protocol("WM_DELETE_WINDOW", close_window)
    def key(code):
        keys = {67:"close", 79:"open", 82:"reset", 32:"pause", 68:"cycle"}
        if code in keys:
            command(keys[code])
        if code in (49, 50, 51, 52):
            set_view(("grip", "drop", "robot", "room")[code-49])
    print("G1 grip demo ready | O open, C close, R reload, D demo, Space pause, 1-4 cameras", flush=True)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key) as view:
        while view.is_running() and state["alive"]:
            started = time.perf_counter()
            window.update_idletasks()
            window.update()
            action, state["action"] = state["action"], None
            if action == "pause":
                state["paused"] = not state["paused"]
            elif action in ("open", "close"):
                sim.closed = action == "close"
                state["cycle_release"] = None
            elif action in ("reset", "default", "cycle"):
                if action == "default":
                    sim.pose = dict(POSE)
                    for n, var in variables.items():
                        var.set(math.degrees(POSE[n]))
                sim.reset()
                state["paused"] = False
                state["cycle_release"] = 3. if action == "cycle" else None
            if state["dirty_camera"]:
                cam = camera(sim, state["camera"])
                view.cam.type = cam.type
                view.cam.lookat[:] = cam.lookat
                view.cam.distance, view.cam.azimuth, view.cam.elevation = cam.distance, cam.azimuth, cam.elevation
                state["dirty_camera"] = False
            if not state["paused"]:
                for _ in range(round(1/60/sim.model.opt.timestep)):
                    if state["cycle_release"] is not None and sim.data.time >= state["cycle_release"]:
                        sim.closed = False
                        state["cycle_release"] = None
                        set_view("drop")
                    sim.step()
            sim.update_guide()
            sample = sim.sample()
            status.configure(text=f"{'PAUSED' if state['paused'] else 'RUNNING'} | {'CLOSED' if sim.closed else 'OPEN'} | sim {sim.data.time:.2f} s\n"
                f"Contact normals: {sample['thumb_normal_N']:.3f} / {sample['index_normal_N']:.3f} N\n"
                f"Dart height: {sample['com_z_m']:.3f} m  |  vertical speed: {sample['vz_m_s']:.2f} m/s\n"
                f"Gap: {sample['aperture_m']*1000:.1f} mm  |  aim error: {sample['aim_error_deg']:.2f} deg")
            view.sync()
            time.sleep(max(0, 1/60-(time.perf_counter()-started)))
    window.destroy()
