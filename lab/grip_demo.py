"""Supported G1, manually posed arm, force-limited two-finger grip and gravity release.

Run: python grip_demo.py [--check] [--render]
The dart is always free. Reset is the only operation that places it in the hand.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dart_model import DartDynamics, load_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "grip_demo"
ARM_NAMES = [f"right_{s}_joint" for s in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw")]
# Hand-authored pose. No video reconstruction or learned controller is used.
POSE = dict(zip(ARM_NAMES, [-1.45, -.15, 0., 1.15, .22894461, .00216948, -.15878494]))
POSE.update(left_shoulder_roll_joint=.16, left_elbow_joint=.65)
GRIP_X = .145
CLOSED = .00354  # half aperture, equals the reference barrel radius
OPEN = .036      # 72 mm opening also clears the 42.9 mm flight span


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def vec(a):
    return " ".join(f"{x:.12g}" for x in np.ravel(a))


def build_model(timestep=.0005):
    """Compose a separate demo; do not edit the room or vendored robot model."""
    root = ET.parse(ROOT / "scene.xml").getroot()
    for inc in list(root.findall("include")):
        included = ET.parse(ROOT / inc.attrib["file"]).getroot()
        root.remove(inc)
        for element in included:
            root.append(element)
    root.set("model", "G1 contact grip and gravity release (supported base)")
    compiler = root.find("compiler")
    compiler.set("meshdir", str(ROOT / "assets/g1/meshes"))
    for tex in root.findall(".//texture[@file]"):
        tex.set("file", str(ROOT / tex.attrib["file"]))
    root.find("option").set("timestep", str(timestep))
    root.find("option").set("iterations", "100")
    root.find("option").set("tolerance", "1e-10")
    pelvis = root.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("joint[@name='floating_base_joint']"))
    # Fixed pelvis is a declared fixture, not a balance policy.
    world = root.find("worldbody")
    add(world, "geom", name="support_base", type="box", pos=".07 0 .035",
        size=".24 .27 .035", rgba=".18 .21 .25 1")
    add(world, "geom", name="support_post", type="capsule", fromto=".07 0 .07 .07 0 .79",
        size=".025", rgba=".85 .57 .18 1", contype="0", conaffinity="0")
    add(world, "geom", name="support_brace", type="capsule", fromto=".07 0 .79 .30 0 .79",
        size=".018", rgba=".85 .57 .18 1", contype="0", conaffinity="0")
    hand = root.find(".//body[@name='right_wrist_yaw_link']")
    for geom in list(hand.findall("geom")):
        if geom.get("mesh") == "right_rubber_hand":
            hand.remove(geom)
    palm = add(hand, "body", name="pinch_palm", pos=".035 0 0")
    add(palm, "geom", name="pinch_palm_geom", type="box", pos=".023 0 0",
        size=".023 .046 .018", mass=".12", rgba=".18 .24 .29 1")
    add(hand, "site", name="pinch_center", pos=f"{GRIP_X} 0 0", size=".002", rgba="0 1 0 0")
    add(hand, "site", name="pinch_forward", pos=f"{GRIP_X+.10} 0 0", size=".003", rgba="0 .7 .9 0")
    actuator = root.find("actuator")
    for side, sign in (("thumb", -1), ("index", 1)):
        # q is half aperture. Each finger travels outward along its own joint axis.
        finger = add(hand, "body", name=f"pinch_{side}")
        add(finger, "joint", name=f"pinch_{side}_joint", type="slide", axis=f"0 {sign} 0",
            range="0 .04", damping="1.2", armature=".002")
        add(finger, "geom", name=f"pinch_{side}_link", type="box",
            pos=f".107 {sign*.009} 0", size=".038 .006 .012", mass=".022",
            rgba=".67 .72 .75 1")
        add(finger, "geom", name=f"pinch_{side}_pad", type="box",
            pos=f"{GRIP_X} {sign*.003} 0", size=".012 .003 .013", mass=".006",
            rgba=".12 .32 .36 1", friction=".7 .002 .00005", condim="6", priority="2",
            solref=".003 1", solimp=".95 .99 .001", margin=".00005")
        add(actuator, "motor", name=f"pinch_{side}_motor", joint=f"pinch_{side}_joint",
            ctrllimited="true", ctrlrange="-4 4", forcelimited="true", forcerange="-4 4")
    # A distinctive physical landing mat catches the dart; no scripted attachment.
    add(world, "geom", name="drop_mat", type="box", pos=".88 -.08 .012",
        size=".28 .25 .012", rgba=".11 .30 .32 1", friction=".8 .002 .00005")
    # Geometry used only as a visual aim guide, never as a collision object.
    add(world, "geom", name="aim_guide", type="capsule", fromto=".9 0 1.2 2.99 0 1.73",
        size=".0015", rgba=".18 .85 .9 .28", contype="0", conaffinity="0", mass="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class GripDemo:
    def __init__(self, timestep=.0005):
        self.model = build_model(timestep)
        self.data = mujoco.MjData(self.model)
        self.dart = self.model.body("spare_dart_1").id
        self.dart_q = int(self.model.joint("spare_dart_1_free").qposadr[0])
        self.dart_v = int(self.model.joint("spare_dart_1_free").dofadr[0])
        self.hand = self.model.body("right_wrist_yaw_link").id
        self.site = self.model.site("pinch_center").id
        self.finger_joints = [self.model.joint(f"pinch_{s}_joint").id for s in ("thumb", "index")]
        self.finger_act = [self.model.actuator(f"pinch_{s}_motor").id for s in ("thumb", "index")]
        self.robot_act = [i for i in range(self.model.nu) if i not in self.finger_act]
        joints = self.model.actuator_trnid[self.robot_act, 0]
        self.robot_q = self.model.jnt_qposadr[joints]
        self.robot_v = self.model.jnt_dofadr[joints]
        wrist = np.array(["wrist" in self.model.joint(j).name for j in joints])
        self.kp = np.where(wrist, 100., 180.)
        self.kd = np.where(wrist, 4., 9.)
        self.dynamics = DartDynamics(self.model)
        self.force = .8  # commanded inward force in N per finger
        self.closed = True
        self.pose = dict(POSE)
        self.reset()

    def reset(self, closed=True):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for name, value in self.pose.items():
            d.qpos[m.joint(name).qposadr[0]] = value
        for jid in self.finger_joints:
            d.qpos[m.jnt_qposadr[jid]] = CLOSED if closed else OPEN
        mujoco.mj_forward(m, d)
        d.qpos[self.dart_q:self.dart_q+3] = d.site_xpos[self.site]
        # Dart local +X is the tip: +90 degrees about world Y makes it point down.
        d.qpos[self.dart_q+3:self.dart_q+7] = [math.sqrt(.5), 0, math.sqrt(.5), 0]
        self.target = d.qpos.copy()
        self.closed = closed
        mujoco.mj_forward(m, d)
        self.initial_z = float(d.xipos[self.dart, 2])
        self.update_guide()

    def update_guide(self):
        m, d = self.model, self.data
        gid = m.geom("aim_guide").id
        start = d.site_xpos[self.site].copy()
        end = d.site_xpos[m.site("bullseye").id].copy()
        delta = end-start
        m.geom_pos[gid] = (start+end)/2
        m.geom_size[gid, 1] = np.linalg.norm(delta)/2
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, delta)
        m.geom_quat[gid] = quat

    def set_joint(self, name, value):
        jid = self.model.joint(name).id
        value = float(np.clip(value, *self.model.jnt_range[jid]))
        self.pose[name] = value
        self.target[self.model.jnt_qposadr[jid]] = value

    def step(self):
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        # Torque limited joint servos with gravity compensation, not qpos playback.
        torque = (self.kp*(self.target[self.robot_q]-d.qpos[self.robot_q])
                  -self.kd*d.qvel[self.robot_v]+d.qfrc_bias[self.robot_v])
        limits = m.actuator_ctrlrange[self.robot_act]
        d.ctrl[self.robot_act] = np.clip(torque, limits[:, 0], limits[:, 1])
        for aid, jid in zip(self.finger_act, self.finger_joints):
            q, v = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
            if self.closed:
                command = -self.force - .25*d.qvel[v]
            else:
                command = 220*(OPEN-d.qpos[q])-1.5*d.qvel[v]
            d.ctrl[aid] = np.clip(command, -4, 4)
        self.dynamics.apply(d)
        mujoco.mj_step(m, d)

    def sample(self):
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        forces = {"thumb": 0., "index": 0.}
        vertical = 0.
        other_contacts = []
        contact_force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.dart not in (b1, b2):
                continue
            other = c.geom2 if b1 == self.dart else c.geom1
            name = m.geom(other).name
            mujoco.mj_contactForce(m, d, i, contact_force)
            # Contact wrench is on geom2; rotate contact coordinates into world.
            world_force = c.frame.reshape(3, 3).T @ contact_force[:3]
            vertical += world_force[2] * (1 if b2 == self.dart else -1)
            if name in ("pinch_thumb_pad", "pinch_index_pad"):
                forces["thumb" if "thumb" in name else "index"] += float(contact_force[0])
            else:
                other_contacts.append(name)
        aim = d.site_xpos[m.site("bullseye").id]-d.site_xpos[self.site]
        axis = d.xmat[self.hand].reshape(3, 3)[:, 0]
        angle = math.degrees(math.acos(float(np.clip(axis @ aim/np.linalg.norm(aim), -1, 1))))
        return {"time_s": float(d.time), "com_z_m": float(d.xipos[self.dart, 2]),
                "vz_m_s": float(d.qvel[self.dart_v+2]),
                "thumb_normal_N": forces["thumb"], "index_normal_N": forces["index"],
                "contact_vertical_N": float(vertical), "other_dart_contacts": other_contacts,
                "aperture_m": float(sum(d.qpos[m.jnt_qposadr[j]] for j in self.finger_joints)),
                "aim_error_deg": angle, "closed": self.closed,
                "tip_down_cosine": float(-d.xmat[self.dart].reshape(3, 3)[2, 0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.check:
        from grip_demo_tools import validate
        validate()
    elif args.render:
        from grip_demo_tools import render
        render()
    else:
        from grip_demo_tools import viewer
        viewer()


if __name__ == "__main__":
    main()
