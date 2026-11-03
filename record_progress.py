"""Record three application-window videos of the current MuJoCo progress.

The visible Tk frontend shows MuJoCo rendering and real joint-control widgets.
FFmpeg gdigrab records only this window. Backend keyframes drive the same servo
targets used by the interactive grip demo; the dart always uses simulation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import tkinter as tk
from tkinter import ttk

import mujoco
import numpy as np
from PIL import Image, ImageTk

from grip_demo import ARM_NAMES, GripDemo, POSE, ROOT
from grip_demo_tools import advance
from view_scene import load_scene

OUTPUT = ROOT / "progress_videos"
TITLE = "G1 Darts - Progress Recording"


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x*x*(3-2*x)


def blend(a, b, u):
    return np.asarray(a)*(1-smooth(u))+np.asarray(b)*smooth(u)


class Frontend:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.geometry("1440x880+40+40")
        self.root.resizable(False, False)
        self.root.configure(bg="#12212c")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#12212c")
        style.configure("TLabel", background="#12212c", foreground="#d9e6e9", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"), foreground="#f5f5ed")
        style.configure("Accent.TLabel", foreground="#6ee2d1", font=("Segoe UI", 12, "bold"))
        self.header = ttk.Label(self.root, text="G1 DARTS LAB", style="Title.TLabel")
        self.header.place(x=24, y=15)
        self.phase = ttk.Label(self.root, text="", style="Accent.TLabel")
        self.phase.place(x=24, y=53)
        self.image = tk.Label(self.root, bg="#12212c", bd=0)
        self.image.place(x=0, y=93, width=1100, height=720)
        self.panel = ttk.Frame(self.root, padding=12)
        self.panel.place(x=1110, y=93, width=320, height=720)
        self.footer = ttk.Label(self.root, text="MuJoCo | current project progress", font=("Segoe UI", 11))
        self.footer.place(x=24, y=835)
        self.detail = ttk.Label(self.panel, text="", justify="left", wraplength=290)
        self.detail.pack(anchor="w", pady=(0, 10))
        self.sim = None
        self.slider_vars = {}
        self.value_labels = {}
        ttk.Label(self.panel, text="ARM JOINT TARGETS", style="Accent.TLabel").pack(anchor="w", pady=(5, 10))
        for name in ARM_NAMES:
            ttk.Label(self.panel, text=name.replace("right_", "").replace("_joint", "").replace("_", " ")).pack(anchor="w")
            row = ttk.Frame(self.panel)
            row.pack(fill="x", pady=(0, 3))
            var = tk.DoubleVar(value=math.degrees(POSE[name]))
            self.slider_vars[name] = var
            label = ttk.Label(row, text=f"{var.get():.1f} deg", width=10)
            label.pack(side="right")
            self.value_labels[name] = label
            ttk.Scale(row, from_=-120 if "shoulder" in name else -90,
                      to=120, variable=var,
                      command=lambda v, n=name: self.target(n, math.radians(float(v)))).pack(fill="x", expand=True)
        self.force_var = tk.DoubleVar(value=.8)
        ttk.Label(self.panel, text="INWARD GRIP FORCE / FINGER", style="Accent.TLabel").pack(anchor="w", pady=(10, 4))
        self.force_label = ttk.Label(self.panel, text="0.80 N")
        self.force_label.pack(anchor="w")
        ttk.Scale(self.panel, from_=0, to=2, variable=self.force_var, command=self.set_force).pack(fill="x")
        buttons = ttk.Frame(self.panel)
        buttons.pack(fill="x", pady=9)
        ttk.Button(buttons, text="Close", command=lambda:self.grip(True)).pack(side="left", padx=2)
        ttk.Button(buttons, text="Open / drop", command=lambda:self.grip(False)).pack(side="left", padx=2)
        ttk.Button(buttons, text="Reload", command=lambda:self.sim.reset() if self.sim else None).pack(side="left", padx=2)
        self.telemetry = ttk.Label(self.panel, text="", justify="left", font=("Segoe UI", 10))
        self.telemetry.pack(anchor="w", pady=8)
        self.root.update()

    def target(self, name, value):
        if self.sim:
            self.sim.set_joint(name, value)
        self.slider_vars[name].set(math.degrees(value))
        self.value_labels[name].configure(text=f"{math.degrees(value):.1f} deg")

    def set_force(self, value):
        if self.sim:
            self.sim.force = float(value)
        self.force_var.set(float(value))
        self.force_label.configure(text=f"{float(value):.2f} N")

    def grip(self, closed):
        if self.sim:
            self.sim.closed = closed

    def show(self, renderer, data, cam):
        renderer.update_scene(data, camera=cam)
        self.photo = ImageTk.PhotoImage(Image.fromarray(renderer.render()))
        self.image.configure(image=self.photo)
        if self.sim:
            s = self.sim.sample()
            self.telemetry.configure(text=(f"{'GRIP CLOSED' if self.sim.closed else 'GRIP OPEN'} | sim {data.time:.2f} s\n"
                f"Contact: {s['thumb_normal_N']:.3f} / {s['index_normal_N']:.3f} N\n"
                f"Gap: {s['aperture_m']*1000:.1f} mm\n"
                f"Dart height: {s['com_z_m']:.3f} m\n"
                f"Vertical speed: {s['vz_m_s']:.2f} m/s"))
        self.root.update_idletasks()
        self.root.update()


def free_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def tour_camera(t):
    # Camera orientation is continuous within each architectural detail shot.
    if t < 8:
        return free_camera([1.05, 0, 1.1], 5.0-.055*t, 40+4*t, -19), "ROOM / orbit and throwing lane"
    if t < 13:
        u = (t-8)/5
        return free_camera([2.99, 0, 1.73], 1.25-.22*u, -8+16*u, -3), "DARTBOARD / bull at 1.73 m"
    if t < 18:
        u = (t-13)/5
        return free_camera([.46, -1.2, .83], .65-.1*u, 100+40*u, -25), "DART ASSET / barrel, stem and flights"
    return free_camera([1.2, 0, 1.1], 4.4, 55+3*(t-18), -16), "ENVIRONMENT / robot, oche, board and equipment"


def record(kind):
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "_qa").mkdir(exist_ok=True)
    app = Frontend()
    if kind == "room":
        model, data = load_scene()
        duration, filename = 23, "01_room_and_environment.mp4"
        app.header.configure(text="G1 DARTS LAB / ENVIRONMENT TOUR")
        app.detail.configure(text="ROOM & ASSETS\n\nBull height: 1.73 m\nOche to board: 2.37 m\n\nG1 humanoid\nEquipment table\nReference dart: 26.7 g\n\nInspection pose\nPhysics paused for this tour")
        for child in app.panel.winfo_children()[1:]:
            child.pack_forget()
        app.footer.configure(text="Environment inspection | Camera orbit and close-ups | MuJoCo frontend capture")
    else:
        app.sim = GripDemo()
        model, data = app.sim.model, app.sim.data
        duration = 25 if kind == "controls" else 17
        filename = "02_joint_and_grip_controls.mp4" if kind == "controls" else "03_physical_hold_and_release.mp4"
        app.header.configure(text="G1 DARTS LAB / "+("JOINT & GRIP CONTROLS" if kind == "controls" else "CONTACT HOLD & GRAVITY RELEASE"))
        app.detail.configure(text="MANUAL JOINT POSE\nCustom two-finger gripper\nSupported robot base\n26.7 g free-body dart")
        app.footer.configure(text="Live physics at 0.5x speed | Torque-limited joints | Contact friction holds the dart | Gravity release")
        if kind == "controls":
            app.sim.reset(closed=False)
            advance(app.sim, 1.3)
        else:
            advance(app.sim, .5)
    path = OUTPUT / filename
    initial_cam = tour_camera(0)[0] if kind == "room" else free_camera([.65, -.04, .90], 2.2, 120, -13)
    with mujoco.Renderer(model, height=720, width=1100) as renderer:
        app.show(renderer, data, initial_cam)
        # Record only this named application window. No desktop or other windows.
        log = (OUTPUT / "_qa" / (filename+".log")).open("w", encoding="utf-8")
        recorder = subprocess.Popen(["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "gdigrab", "-framerate", "25", "-draw_mouse", "0", "-i", "title="+TITLE,
            "-t", str(duration), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdout=log, stderr=log)
        start = time.perf_counter()
        last, sim_target, released, reloaded = 0., float(data.time), False, False
        next_progress = 5.
        while True:
            t = time.perf_counter()-start
            if t >= duration+1 or recorder.poll() is not None:
                break
            if kind == "room":
                cam, phase = tour_camera(t)
            elif kind == "controls":
                phase = "EDIT JOINTS / arm targets drive the physical servos"
                # Deterministic target edits also move the real frontend sliders.
                if t < 4:
                    u = smooth(t/4)
                    app.target(ARM_NAMES[0], -1.45+.38*u)
                    app.target(ARM_NAMES[3], 1.15-.45*u)
                elif t < 8:
                    u = smooth((t-4)/4)
                    app.target(ARM_NAMES[0], -1.07-.38*u)
                    app.target(ARM_NAMES[3], .70+.45*u)
                elif t < 12:
                    app.target(ARM_NAMES[4], POSE[ARM_NAMES[4]]+.25*math.sin((t-8)*math.pi/2))
                    phase = "EDIT WRIST / roll adjustment"
                elif t < 17:
                    app.sim.closed = t < 14.5
                    app.set_force(.4+.8*smooth((t-12)/5))
                    phase = "GRIP / close, open and adjust inward force"
                else:
                    if not reloaded:
                        for n in ARM_NAMES:
                            app.target(n, POSE[n])
                        app.set_force(.8)
                        app.sim.reset()
                        sim_target = 0.
                        reloaded = True
                    phase = "RELOAD / dart placed tip-down and held by friction"
                cam = free_camera(app.sim.data.site_xpos[app.sim.site], .48, 125, -13) if t >= 12 else free_camera([.62, -.03, .88], 2.1, 115+1.5*t, -13)
            else:
                if t < 5:
                    cam = free_camera([1.25, 0, 1.12], 4.0, 45+2*t, -14)
                    phase = "HOLD / arm points toward bullseye; dart points down"
                elif t < 9:
                    cam = free_camera(data.site_xpos[app.sim.site], .38, 125, -15)
                    phase = "HOLD / 0.8 N normal force per pad"
                else:
                    if not released:
                        app.sim.closed = False
                        released = True
                    cam = free_camera([.84, -.06, .68], 1.6, 110, -12)
                    phase = "OPEN / free dart falls under gravity" if t < 12 else "LANDED / physical contact with the mat"
            if app.sim:
                # Fixed-step physics; cap scheduling drift, never move the dart kinematically.
                delta = min(t-last, .1)
                sim_target += .5*delta
                while data.time < sim_target-1e-9:
                    app.sim.step()
                app.sim.update_guide()
            last = t
            app.phase.configure(text=phase)
            app.show(renderer, data, cam)
            if t >= next_progress:
                print(f"{kind}: {t:.1f}/{duration} s", flush=True)
                next_progress += 5
            time.sleep(.005)
        result = recorder.wait(timeout=20)
        log.close()
        if result:
            raise RuntimeError(f"Recording failed: {path}; see log")
    app.root.destroy()
    print(path, flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", choices=("room", "controls", "release", "all"), default="all")
    args = parser.parse_args()
    for clip in (("room", "controls", "release") if args.clip == "all" else (args.clip,)):
        record(clip)
