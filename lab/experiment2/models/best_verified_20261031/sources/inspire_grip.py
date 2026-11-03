"""G1 Inspire five-finger tripod grasp. Free dart; torque actuation; supported G1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from dart_model import DartDynamics
from grip_demo import ROOT

OUT = ROOT / "inspire_grip"
from inspire_model import build_model, NAMES, DISTAL


class InspireGrip:
    def __init__(self, checkpoint=None, timestep=.0005):
        self.model = m = build_model(timestep)
        self.data = mujoco.MjData(m)
        self.dart = m.body("spare_dart_1").id
        self.dart_q = int(m.joint("spare_dart_1_free").qposadr[0])
        self.dart_v = int(m.joint("spare_dart_1_free").dofadr[0])
        self.hand = m.body("left_wrist_yaw_link").id
        self.fj = np.array([m.joint(n).id for n in NAMES])
        self.fq, self.fv = m.jnt_qposadr[self.fj], m.jnt_dofadr[self.fj]
        self.fa = np.array([m.actuator(n).id for n in NAMES])
        self.ra = np.array([i for i in range(m.nu) if i not in self.fa])
        rj = m.actuator_trnid[self.ra, 0]
        self.rq, self.rv = m.jnt_qposadr[rj], m.jnt_dofadr[rj]
        fingers = np.array([m.joint(j).name.startswith("R_") for j in rj])
        wrists = np.array(["wrist" in m.joint(j).name for j in rj])
        self.kp = np.where(fingers, 3., np.where(wrists, 100., 180.))
        self.kd = np.where(fingers, .08, np.where(wrists, 4., 9.))
        self.distal = [m.body(n).id for n in DISTAL]
        self.dynamics = DartDynamics(m)
        checkpoint = OUT / "policy.json" if checkpoint is None else Path(checkpoint)
        self.config = json.loads(checkpoint.read_text())
        if self.config.get("hand") != "left":
            raise ValueError("This demo requires a left-hand checkpoint")
        self.closed = True
        self.hand_enabled = True
        self.reset()

    def place(self, q, pos, quat=None):
        """Reset/IK only. Never called by the dynamics controller."""
        m, d = self.model, self.data
        d.qpos[self.fq] = q
        self.set_coupled()
        mujoco.mj_forward(m, d)
        rot = d.xmat[self.hand].reshape(3, 3)
        d.qpos[self.dart_q:self.dart_q+3] = d.xpos[self.hand] + rot @ pos
        local = np.asarray(quat if quat is not None else self.config["dart_quat_hand"])
        wquat = np.zeros(4)
        mujoco.mju_mulQuat(wquat, d.xquat[self.hand], local)
        d.qpos[self.dart_q+3:self.dart_q+7] = wquat
        mujoco.mj_forward(m, d)

    def set_coupled(self):
        m, d = self.model, self.data
        for e in range(m.neq):
            j1,j2=m.eq_obj1id[e],m.eq_obj2id[e]
            d.qpos[m.jnt_qposadr[j1]]=m.eq_data[e,0]+m.eq_data[e,1]*d.qpos[m.jnt_qposadr[j2]]

    def reset(self, perturb=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for name, value in self.config["arm_pose"].items():
            d.qpos[m.joint(name).qposadr[0]] = value
        self.place(self.config["q_grasp"], self.config["dart_pos_hand"])
        if perturb is not None:
            d.qpos[self.dart_q:self.dart_q+3] += perturb
        self.target = d.qpos.copy()
        self.closed = True
        mujoco.mj_forward(m, d)
        self.initial_com = d.xipos[self.dart].copy()
        self.initial_relative = self.relative_pos()
        self.command = d.qpos[self.fq].copy()
        self.force_offset = np.zeros(6)
        self.tick = 0

    def relative_pos(self):
        d = self.data
        return d.xmat[self.hand].reshape(3, 3).T @ (d.xpos[self.dart]-d.xpos[self.hand])

    def step(self):
        m, d = self.model, self.data
        velocity = getattr(self, "velocity_target", np.zeros(m.nv))
        feedforward = d.qfrc_bias.copy()
        if hasattr(self, "acceleration_target"):
            inertial = np.zeros(m.nv)
            mujoco.mj_mulM(m, d, inertial, self.acceleration_target)
            feedforward += inertial
        torque = (self.kp*(self.target[self.rq]-d.qpos[self.rq])
                  +self.kd*(velocity[self.rv]-d.qvel[self.rv])+feedforward[self.rv])
        d.ctrl[self.ra] = np.clip(torque, m.actuator_ctrlrange[self.ra, 0], m.actuator_ctrlrange[self.ra, 1])
        qtarget = np.asarray(self.config["q_grasp"], dtype=float).copy()
        if self.closed:
            qtarget += np.asarray(self.config["closure_direction"]) * self.config["squeeze"]
            if self.config.get("force_feedback", False):
                if self.tick % 10 == 0:
                    measured = np.asarray(self.sample()["normal_N"])
                    error = np.asarray(self.config.get("force_targets", [2.,1.,1.]))-measured
                    self.force_offset[1:4] += np.array([-1.,1.,1.]) * error * self.config.get("force_gain", .04) * (10*m.opt.timestep)
                    self.force_offset[1:4] = np.clip(self.force_offset[1:4], -.12, .12)
                qtarget += self.force_offset
        else:
            qtarget = np.asarray(self.config["q_open"])
        qtarget = np.clip(qtarget, m.jnt_range[self.fj, 0], m.jnt_range[self.fj, 1])
        # Respect URDF joint velocity limits in reference generation, including thumb coupling.
        rate = np.array([.5,.5/2.4,.5,.5,.5,.5])
        self.command += np.clip(qtarget-self.command, -rate*m.opt.timestep, rate*m.opt.timestep)
        command = self.config["kp"]*(self.command-d.qpos[self.fq])-self.config["kd"]*d.qvel[self.fv]+d.qfrc_bias[self.fv]
        # Conservative software ceiling below manufacturer motor torque limits.
        d.ctrl[self.fa] = np.clip(command, -.35, .35) if self.hand_enabled else 0.
        self.dynamics.apply(d)
        mujoco.mj_step(m, d)
        self.tick += 1

    def sample(self):
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        forces = np.zeros(3)
        vertical = 0.
        unwanted = []
        regions = [set(), set(), set()]
        penetration = 0.
        contact = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
            if self.dart not in (b1, b2):
                continue
            other = b2 if b1 == self.dart else b1
            mujoco.mj_contactForce(m, d, i, contact)
            if contact[0] <= 1e-5:
                continue
            vertical += float((c.frame.reshape(3, 3).T @ contact[:3])[2]) * (1 if b2 == self.dart else -1)
            penetration = max(penetration, -float(c.dist))
            if other in self.distal:
                index = self.distal.index(other)
                forces[index] += contact[0]
                dart_geom = m.geom(c.geom1 if b1 == self.dart else c.geom2).name
                regions[index].add(dart_geom)
                if "_barrel_" not in dart_geom:
                    unwanted.append(m.body(other).name + ":" + dart_geom)
            else:
                unwanted.append(m.body(other).name)
        return {"time_s": float(d.time), "normal_N": forces.tolist(),
                "contact_regions": [sorted(region) for region in regions],
                "vertical_contact_N": vertical, "other_contacts": sorted(set(unwanted)),
                "penetration_m": penetration, "com": d.xipos[self.dart].tolist(),
                "relative_drift_m": float(np.linalg.norm(self.relative_pos()-self.initial_relative)),
                "vz": float(d.qvel[self.dart_v+2]), "closed": self.closed,
                "q": d.qpos[self.fq].tolist(), "torque_Nm": d.ctrl[self.fa].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUT / "policy.json")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--fit", action="store_true", help="Refit the photo-guided contact pose")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    from inspire_grip_tools import fit_pose, validate, render, viewer
    if args.fit:
        fit_pose()
    elif args.train:
        from train_inspire_grip import train
        train()
    elif args.check:
        validate(args.checkpoint)
    elif args.render:
        render(args.checkpoint)
    else:
        viewer(args.checkpoint)


if __name__ == "__main__":
    main()
