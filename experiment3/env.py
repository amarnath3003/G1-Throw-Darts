"""Step 1: supported G1, empty left hand, free tabletop dart. No grasp reset."""
from pathlib import Path
import json
import hashlib

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ARM = ['left_'+n+'_joint' for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw',
                                 'elbow','wrist_roll','wrist_pitch','wrist_yaw')]
FINGERS = ['L_'+n+'_joint' for n in ('thumb_proximal_yaw','thumb_proximal_pitch',
                                  'index_proximal','middle_proximal','ring_proximal','pinky_proximal')]


def geom_z_bounds(m, d, g):
    """Exact vertical support extrema for collision primitives/mesh vertices."""
    r = d.geom_xmat[g].reshape(3,3)
    kind = m.geom_type[g]
    size = m.geom_size[g]
    z = d.geom_xpos[g,2]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        heights = m.mesh_vert[start:start+count] @ r[2] + z
        return float(heights.min()), float(heights.max())
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[1]*abs(r[2,2]) + size[0]*np.linalg.norm(r[2,:2])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[1]*abs(r[2,2]) + size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(r[2]) @ size
    else:
        raise ValueError(f'Unsupported geom type {kind}: {m.geom(g).name}')
    return float(z-extent), float(z+extent)


class PickupScene:
    def __init__(self, timestep=None):
        self.config = json.loads((HERE/'configs/scene.json').read_text())
        path = HERE/'configs/scene.mjb'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.config['model_sha256']:
            raise ValueError('Model snapshot hash mismatch')
        grasp_path = HERE/'configs/grasp_target.json'
        if hashlib.sha256(grasp_path.read_bytes()).hexdigest() != self.config['source_sha256']['inspire_grip/policy.json']:
            raise ValueError('Grasp snapshot hash mismatch')
        self.grasp = json.loads(grasp_path.read_text())
        self.model = m = mujoco.MjModel.from_binary_path(str(path))
        if timestep is not None:
            m.opt.timestep = timestep
        self.data = mujoco.MjData(m)
        self.dart = m.body('spare_dart_1').id
        self.hand = m.body('left_wrist_yaw_link').id
        self.dq = int(m.joint('spare_dart_1_free').qposadr[0])
        self.dv = int(m.joint('spare_dart_1_free').dofadr[0])
        self.arm_ids = np.array([m.joint(n).id for n in ARM])
        self.aq = m.jnt_qposadr[self.arm_ids]
        self.fq = np.array([m.joint(n).qposadr[0] for n in FINGERS])
        self.dart_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.dart]
        self.act_joints = m.actuator_trnid[:,0]
        self.rq = m.jnt_qposadr[self.act_joints]
        self.rv = m.jnt_dofadr[self.act_joints]
        finger = np.array([(m.joint(j).name or '').startswith(('L_','R_')) for j in self.act_joints])
        wrist = np.array(['wrist' in m.joint(j).name for j in self.act_joints])
        self.kp = np.where(finger, self.grasp['kp'], np.where(wrist,100.,180.))
        self.kd = np.where(finger, self.grasp['kd'], np.where(wrist,4.,9.))
        self.limit = np.minimum(np.abs(m.actuator_ctrlrange[:,1]), np.where(finger,.35,np.inf))
        self.reset()

    def couple(self):
        m,d = self.model,self.data
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
                a,b = m.eq_obj1id[e],m.eq_obj2id[e]
                d.qpos[m.jnt_qposadr[a]] = np.polynomial.polynomial.polyval(d.qpos[m.jnt_qposadr[b]],m.eq_data[e,:5])

    def reset(self):
        m,d = self.model,self.data
        mujoco.mj_resetData(m,d)
        for name,value in self.grasp['arm_pose'].items():
            d.qpos[m.joint(name).qposadr[0]] = value
        d.qpos[self.fq] = self.grasp['q_open']
        self.couple()
        # Initialization only: free dart placed just above the surface using all geometry.
        d.qpos[self.dq:self.dq+3] = [*self.config['dart_xy'],0.]
        d.qpos[self.dq+3:self.dq+7] = self.config['dart_quat_wxyz']
        mujoco.mj_forward(m,d)
        lowest = min(geom_z_bounds(m,d,g)[0] for g in self.dart_geoms)
        d.qpos[self.dq+2] = self.config['table_top_z'] - lowest + .001
        self.target = d.qpos.copy()
        mujoco.mj_forward(m,d)
        for _ in range(round(self.config['settle_seconds']/m.opt.timestep)):
            self.step()
        # Preserve physically settled positions/velocities. No object edits after settling.
        return self.data.qpos.copy()

    def step(self):
        m,d = self.model,self.data
        torque = self.kp*(self.target[self.rq]-d.qpos[self.rq])-self.kd*d.qvel[self.rv]+d.qfrc_bias[self.rv]
        d.ctrl[:] = np.clip(torque,-self.limit,self.limit)
        # Air loads are omitted in this stationary setup audit; restore pinned DartDynamics for transport.
        mujoco.mj_step(m,d)

    def contacts(self):
        mujoco.mj_forward(self.model,self.data)
        result=[]
        for c in self.data.contact:
            result.append(dict(geom1=self.model.geom(c.geom1).name,
                               geom2=self.model.geom(c.geom2).name, distance_m=float(c.dist)))
        return result


if __name__ == '__main__':
    import time
    import mujoco.viewer
    scene = PickupScene()
    with mujoco.viewer.launch_passive(scene.model,scene.data) as viewer:
        viewer.cam.lookat[:] = [.43,.34,.96]
        viewer.cam.distance = 1.55
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        while viewer.is_running():
            start = time.perf_counter()
            for _ in range(40):
                scene.step()
            viewer.sync()
            time.sleep(max(0.,40*scene.model.opt.timestep-(time.perf_counter()-start)))
