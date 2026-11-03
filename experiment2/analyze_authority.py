"""Local kinematic velocity bounds at recorded poses; not a success simulator."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from env import DartThrowEnv


def analyze(folder,variant):
    states=np.load(folder/'states.npz');result=json.loads((folder/'result.json').read_text())
    env=DartThrowEnv(variant=variant);m,d=env.m,env.d
    urdf=ET.parse(Path(__file__).resolve().parents[1]/'assets/inspire/upstream_g1.urdf').getroot()
    hardware=np.array([float(urdf.find(f"joint[@name='{name}']/limit").get('velocity')) for name in env.names])
    jp=np.zeros((3,m.nv));jr=jp.copy();rows=[]
    for requested in (.04,result['release']['time_s']):
        index=int(np.argmin(abs(states['time']-requested)))
        d.qpos[:]=states['qpos'][index];d.qvel[:]=states['qvel'][index];env.t=float(states['time'][index]);mujoco.mj_forward(m,d)
        point=d.xipos[env.sim.dart].copy();mujoco.mj_jac(m,d,jp,jr,point,env.sim.hand)
        J=jp[:,env.vadr];other=jp@d.qvel-J@d.qvel[env.vadr]
        nominal=env.curve(min(env.t+2.1,3.8),1)[env.qadr]
        rate=np.broadcast_to(np.asarray(env.slew_limit())/.02,(9,))
        delta=env.goal-point;fits={}
        for label,lower,upper in [('reference_rate',np.maximum(nominal-rate,-hardware),np.minimum(nominal+rate,hardware)),('urdf_speed_only',-hardware,hardware)]:
            best=None
            for T in np.linspace(.2,1.2,51):
                wanted=delta/T+np.array([0,0,4.905*T])
                fit=lsq_linear(J,wanted-other,bounds=(lower,upper))
                error=float(np.linalg.norm(J@fit.x+other-wanted))
                if best is None or error<best['velocity_error_m_s']:
                    best={'velocity_error_m_s':error,'flight_time_s':float(T),'desired_velocity':wanted.tolist(),
                          'achievable_velocity':(J@fit.x+other).tolist(),'joint_rates':fit.x.tolist()}
            fits[label]=best
        rows.append({'time_s':env.t,'hand_point_jacobian':J.tolist(),'measured_joint_rates':d.qvel[env.vadr].tolist(),'fits':fits})
    env.close()
    return {'folder':str(folder),'variant':variant,'joint_names':env.names,'poses':rows,
            'limitation':'Local rigid-hand Jacobian bounds ignore contact retention, torque/acceleration limits, tracking error and changing future poses. They neither prove a successful throw nor global task impossibility.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--variant',default='arc_v5')
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    report=analyze(a.folder,a.variant);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps([{ 'time_s':r['time_s'],'fits':r['fits']} for r in report['poses']],indent=2))


if __name__=='__main__':main()
