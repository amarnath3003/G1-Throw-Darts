"""Independent invariants, aerodynamic signs, reference stability and flight convergence."""
from copy import deepcopy
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from dart_model import DartDynamics, load_profile, properties, sample_parameters, validate_profile

ROOT=Path(__file__).resolve().parent
PROFILE=load_profile()
PROP=properties(PROFILE)


def setup(dt=.0005):
    model=mujoco.MjModel.from_xml_path(str(ROOT/"assets/dart.xml"))
    model.opt.timestep=dt
    return model,mujoco.MjData(model)


def flight(dt,air=True):
    model,data=setup(dt)
    # Illustrative benchmark launch, not a trained or measured robot throw.
    pitch=math.radians(19.4)
    data.qpos[3:7]=[math.cos(pitch/2),0,-math.sin(pitch/2),0]
    data.qvel[:3]=[6*math.cos(math.radians(16.5)),0,6*math.sin(math.radians(16.5))]
    data.qvel[3:]=[0,math.radians(412),0]
    mujoco.mj_forward(model,data)
    dyn=DartDynamics(model,("dart",),PROFILE)
    positions=[]
    for _ in range(round(.35/dt)):
        if air:
            dyn.step(data)
        else:
            mujoco.mj_step(model,data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        positions.append(data.qpos[:3].copy())
    assert not any(w.number for w in data.warning)
    return model,data,np.asarray(positions)


def main():
    model,data=setup()
    dyn=DartDynamics(model,("dart",),PROFILE)
    bid=model.body("dart").id
    assert model.nq==7 and model.nv==6
    assert np.isclose(model.body_mass[bid],.0267,rtol=0,atol=1e-12)
    assert np.isclose(PROP["tip_x_m"]-model.body_ipos[bid,0],.0485,atol=1e-12)
    assert np.all(model.body_inertia[bid]>0)
    assert 2*max(model.body_inertia[bid]) <= sum(model.body_inertia[bid])+1e-14
    # Ambient air changes must not change the solid object's inferred inertia.
    p=deepcopy(PROFILE)
    p["air_density_kg_m3"]=1.05
    assert properties(p)["inertia_kg_m2"]==PROP["inertia_kg_m2"]
    p["measured_inertia_kg_m2"]=[3e-7,1.9e-5,1.9e-5]
    validate_profile(p)
    assert properties(p)["inertia_kg_m2"]==p["measured_inertia_kg_m2"]
    force,torque=dyn.wrench(np.eye(3),np.zeros(3),np.zeros(3),model.body_ipos[bid])
    assert np.array_equal(force,np.zeros(3)) and np.array_equal(torque,np.zeros(3))
    # Coordinate/sign checks in yaw and pitch: nose-up/misaligned flight restores.
    eps=1e-6
    r=np.array([[math.cos(eps),0,-math.sin(eps)],[0,1,0],[math.sin(eps),0,math.cos(eps)]])
    _,torque=dyn.wrench(r,np.array([6.,0,0]),np.zeros(3),model.body_ipos[bid])
    ma=-torque[1]/(eps*model.body_inertia[bid,1])
    assert abs(ma/PROFILE["reference_pitch_derivative_s_2"]-1)<1e-4
    r=np.array([[math.cos(eps),-math.sin(eps),0],[math.sin(eps),math.cos(eps),0],[0,0,1]])
    _,torque=dyn.wrench(r,np.array([6.,0,0]),np.zeros(3),model.body_ipos[bid])
    assert torque[2]<0
    _,torque=dyn.wrench(np.eye(3),np.array([6.,0,0]),np.array([0,eps,0]),model.body_ipos[bid])
    mq=torque[1]/(eps*model.body_inertia[bid,1])
    assert abs(mq/PROFILE["reference_pitch_rate_derivative_s_1"]-1)<.002
    # Air must dissipate relative kinetic energy, including the force moment arm.
    rng=np.random.default_rng(73)
    worst_power=-float("inf")
    for _ in range(250):
        velocity=rng.normal(size=3)*7
        omega=rng.normal(size=3)*20
        f,t=dyn.wrench(np.eye(3),velocity,omega,model.body_ipos[bid])
        power=float(f@velocity+t@omega)
        worst_power=max(worst_power,power)
        assert power<=1e-10
    # Body-level wrench application produces the intended translational force.
    data.qvel[0]=6
    mujoco.mj_forward(model,data)
    dyn.apply(data)
    mujoco.mj_forward(model,data)
    expected=data.xfrc_applied[bid,:3]/model.body_mass[bid]+model.opt.gravity
    assert np.allclose(data.qacc[:3],expected,atol=1e-9)
    # No-air ballistic COM motion versus analytic gravity (symplectic integration tolerance).
    m,d=setup()
    initial=d.qpos[:3].copy()
    d.qvel[:3]=[6,0,2]
    for _ in range(600):
        mujoco.mj_step(m,d)
    expected=initial+np.array([6,0,2])*d.time+.5*m.opt.gravity*d.time**2
    ballistic_error=float(np.linalg.norm(d.qpos[:3]-expected))
    assert ballistic_error<.001
    # Two resolutions must agree in impact-relevant position and attitude.
    _,coarse,_=flight(.0005)
    _,fine,_=flight(.00025)
    position_error=float(np.linalg.norm(coarse.qpos[:3]-fine.qpos[:3]))
    orientation_error=2*math.acos(min(1,abs(float(coarse.qpos[3:7]@fine.qpos[3:7]))))
    assert position_error<.001 and orientation_error<math.radians(.25)
    # Independent worker models; randomization cannot leak or drift on repeated resets.
    other,_=setup()
    snapshot=other.body_mass.copy()
    for _ in range(50):
        values=dyn.randomize(data,rng)
        assert abs(model.body_mass[bid]/PROFILE["mass_kg"]-values["mass_scale"])<1e-10
        assert np.allclose(model.body_inertia[bid],np.asarray(PROP["inertia_kg_m2"])*values["mass_scale"]*values["inertia_scale"])
        assert np.array_equal(other.body_mass,snapshot)
        assert abs(model.body_ipos[bid,0]-PROP["com_x_m"])<=.002
        assert np.isfinite(data.qpos).all()
    # Bad profiles must fail before a training run starts.
    bad=deepcopy(PROFILE)
    bad["mass_kg"]=float("nan")
    try:
        validate_profile(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN profile accepted")
    report={"status":"passed","reference":PROFILE["reference"],
            "mass_g":PROFILE["mass_kg"]*1000,"com_behind_tip_mm":PROFILE["com_behind_tip_m"]*1000,
            "inertia_kg_m2":PROP["inertia_kg_m2"],
            "reference_pitch_derivative_s_2":ma,"reference_pitch_rate_derivative_s_1":mq,
            "analytic_ballistic_error_m":ballistic_error,
            "dt_halving_position_error_m":position_error,
            "dt_halving_orientation_error_deg":math.degrees(orientation_error),
            "passive_air_power_checks":250,"randomization_resets_checked":50,
            "transfer_status":"Not validated against project hardware or measured flight trajectories.",
            "limitations":["Drag/contact coefficients and parts of geometry remain assumed.",
                           "Pitch stability agreement is analytical consistency with the reference model, not independent experimental validation.",
                           "Large-angle stall extension, rigid flights, and axial inertia remain approximations."]}
    (ROOT/"dart_validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
