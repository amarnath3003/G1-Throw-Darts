"""Source-backed rigid dart geometry and reusable, explicit aerodynamic dynamics.

Axes: local +X points toward the tip; origin is the barrel midpoint. All units SI.
The XML contains mass/contact properties; air loads require DartDynamics.step().
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROFILE = Path(__file__).with_name("dart_profile.json")


def load_profile(path=PROFILE):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_profile(p)
    return p


def validate_profile(p):
    positive = ("mass_kg", "metal_assembly_mass_kg", "tail_assembly_mass_kg",
                "barrel_length_m", "barrel_diameter_m", "point_exposed_length_m",
                "point_base_diameter_m", "stem_exposed_length_m", "stem_front_diameter_m",
                "stem_rear_diameter_m", "flight_span_m", "flight_root_chord_m",
                "flight_thickness_m", "flight_taper_ratio", "flight_area_m2",
                "mean_aerodynamic_chord_m", "normal_force_slope_per_rad",
                "reference_speed_m_s", "air_density_kg_m3", "stall_angle_rad")
    for key in positive:
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if abs(p["mass_kg"]-p["metal_assembly_mass_kg"]-p["tail_assembly_mass_kg"]) > 1e-9:
        raise ValueError("Component assembly masses must sum to total mass")
    if not 0 < p["com_behind_tip_m"] < p["aerodynamic_center_behind_tip_m"]:
        raise ValueError("Need a measured COM ahead of the aerodynamic center")
    if not np.isfinite(p["reference_pitch_derivative_s_2"]) or p["reference_pitch_derivative_s_2"] >= 0:
        raise ValueError("Reference pitch derivative must be finite and negative")
    if not np.isfinite(p["quarter_chord_sweep_deg"]) or abs(p["quarter_chord_sweep_deg"]) >= 80:
        raise ValueError("Invalid quarter-chord sweep")
    if p["flight_taper_ratio"] > 1:
        raise ValueError("This geometry requires a non-expanding flight tip chord")
    for key in ("axial_drag_coefficient", "crossflow_drag_coefficient", "roll_drag_coefficient"):
        if not np.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"Invalid {key}")
    if np.asarray(p["friction"]).shape != (3,) or not np.isfinite(p["friction"]).all() or min(p["friction"]) < 0:
        raise ValueError("Friction must contain three finite, nonnegative coefficients")
    if np.asarray(p["contact_solref"]).shape != (2,) or not np.isfinite(p["contact_solref"]).all() or min(p["contact_solref"]) <= 0:
        raise ValueError("Contact solref must contain two positive values")
    for key, bounds in p["randomization"].items():
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError(f"Invalid randomization bounds: {key}")
        if key.endswith("scale") and bounds[0] <= 0:
            raise ValueError(f"Randomization scale must be positive: {key}")


def properties(p):
    tip = p["point_exposed_length_m"] + p["barrel_length_m"]/2
    arm = p["aerodynamic_center_behind_tip_m"]-p["com_behind_tip_m"]
    # The reference's analytical pitch derivative fixes transverse inertia independently
    # of the imperfect visible geometry. This reproduces its small-angle stability model.
    iy = (0.5*p["air_density_kg_m3"]*p["reference_speed_m_s"]**2*
          p["flight_area_m2"]*p["normal_force_slope_per_rad"]*arm /
          -p["reference_pitch_derivative_s_2"])
    # Geometry-derived axial inertia: conservative tail split with explicit provenance.
    stem_mass = p["tail_assembly_mass_kg"]*.65
    fin_mass = p["tail_assembly_mass_kg"]-stem_mass
    r = p["flight_span_m"]/2
    taper = p["flight_taper_ratio"]
    mean_r2 = r*r*(1+3*taper)/(6*(1+taper))
    ix = (.5*p["metal_assembly_mass_kg"]*(p["barrel_diameter_m"]/2)**2 +
          .5*stem_mass*(p["stem_front_diameter_m"]/2)**2 + fin_mass*mean_r2)
    return {"mass_kg": p["mass_kg"], "tip_x_m": tip,
            "com_x_m": tip-p["com_behind_tip_m"],
            "aerodynamic_center_x_m": tip-p["aerodynamic_center_behind_tip_m"],
            "static_margin_m": arm, "inertia_kg_m2": [ix, iy, iy]}


def _add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def _v(values):
    return " ".join(f"{float(x):.12g}" for x in np.ravel(values))


def add_assets(asset, p):
    """Closed tapered point, tapered nylon stem, and four swept trapezoidal vanes."""
    def lathe(name, profile):
        vertices, faces = [], []
        n = 64
        for x, r in profile:
            for k in range(n):
                a = 2*math.pi*k/n
                vertices.append([x, r*math.cos(a), r*math.sin(a)])
        for j in range(len(profile)-1):
            for k in range(n):
                a, b = j*n+k, j*n+(k+1)%n
                c, d = (j+1)*n+k, (j+1)*n+(k+1)%n
                faces.extend([[a,b,c], [b,d,c]])
        vertices.extend([[profile[0][0],0,0], [profile[-1][0],0,0]])
        for k in range(n):
            faces.extend([[len(vertices)-2,(k+1)%n,k],
                          [len(vertices)-1,(len(profile)-1)*n+k,(len(profile)-1)*n+(k+1)%n]])
        _add(asset,"mesh",name=name,vertex=_v(vertices),face=_v(faces))
    half = p["barrel_length_m"]/2
    prop = properties(p)
    lathe("dart_point_mesh", [(half,p["point_base_diameter_m"]/2),
                              (prop["tip_x_m"]-0.0001,0.00008), (prop["tip_x_m"],0.00001)])
    lathe("dart_stem_mesh", [(-half-p["stem_exposed_length_m"],p["stem_rear_diameter_m"]/2),
                             (-half,p["stem_front_diameter_m"]/2)])
    # Sweep from reference quarter-chord sweep; position so reference aerodynamic
    # center agrees with quarter-MAC geometry. Root overlaps the nylon stem slot.
    span = p["flight_span_m"]/2
    cr = p["flight_root_chord_m"]
    ct = cr*p["flight_taper_ratio"]
    sweep = math.tan(math.radians(p["quarter_chord_sweep_deg"])) + (cr-ct)/(4*span)
    ym = p["flight_span_m"]/6*(2*p["flight_taper_ratio"]+1)/(p["flight_taper_ratio"]+1)
    root_behind_tip = (p["aerodynamic_center_behind_tip_m"] -
                       p["mean_aerodynamic_chord_m"]/4-ym*sweep)
    front = prop["tip_x_m"]-root_behind_tip
    polygon = [(front,0), (front-span*sweep,span), (front-span*sweep-ct,span), (front-cr,0)]
    thickness = p["flight_thickness_m"]/2
    vertices = [[x,y,z] for z in (-thickness,thickness) for x,y in polygon]
    faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7]]
    for i in range(4):
        j=(i+1)%4
        faces.extend([[i,j,i+4],[j,j+4,i+4]])
    _add(asset,"mesh",name="dart_vane_mesh",vertex=_v(vertices),face=_v(faces))
    for name, rgba, specular, shiny in (
        ("dart_tungsten", ".48 .50 .52 1", ".8", ".55"),
        ("dart_steel", ".67 .70 .74 1", ".95", ".8"),
        ("dart_nylon", ".12 .14 .16 1", ".2", ".15"),
        ("dart_polymer", ".93 .30 .10 1", ".3", ".2"),
    ):
        _add(asset,"material",name=name,rgba=rgba,specular=specular,shininess=shiny)


def add_dart(parent, name, pos, p, free=False, quat="1 0 0 0", color=None):
    prop=properties(p)
    body=_add(parent,"body",name=name,pos=pos,quat=quat)
    if free:
        _add(body,"freejoint",name=f"{name}_free")
    _add(body,"inertial",pos=f"{prop['com_x_m']} 0 0",mass=prop["mass_kg"],diaginertia=_v(prop["inertia_kg_m2"]))
    collision={"mass":"0", "friction":_v(p["friction"]), "solref":_v(p["contact_solref"]),
               "solimp":".95 .99 .001", "condim":"4", "margin":"0.00005"}
    # Axisymmetric grip approximation: dense ring segments create actual shallow
    # grooves rather than oversized painted bands. Maximum radius stays on spec.
    length=p["barrel_length_m"]
    for i in range(24):
        r=p["barrel_diameter_m"]/2*(.95 if i%3 == 1 else 1.)
        x=-length/2+(i+.5)*length/24
        _add(body,"geom",name=f"{name}_barrel_{i}",type="cylinder",pos=f"{x} 0 0",
             size=f"{r} {length/48}",quat=".707106781 0 .707106781 0",material="dart_tungsten",**collision)
    _add(body,"geom",name=f"{name}_point",type="mesh",mesh="dart_point_mesh",material="dart_steel",**collision)
    _add(body,"geom",name=f"{name}_shaft",type="mesh",mesh="dart_stem_mesh",material="dart_nylon",**collision)
    for i in range(4):
        a=math.pi*i/4
        extra={"rgba":color} if color else {}
        _add(body,"geom",name=f"{name}_flight_{i}",type="mesh",mesh="dart_vane_mesh",
             quat=f"{math.cos(a)} {math.sin(a)} 0 0",material="dart_polymer",**collision,**extra)
    for suffix,x,size in (("tip",prop["tip_x_m"],.0005),("grip",0,.004),
                          ("com",prop["com_x_m"],.002),
                          ("aero_center",prop["aerodynamic_center_x_m"],.002)):
        _add(body,"site",name=f"{name}_{suffix}",pos=f"{x} 0 0",size=size,rgba="0 1 0 0")
    return body


def sample_parameters(p, rng):
    bounds=p["randomization"]
    out={key:float(rng.uniform(*value)) for key,value in bounds.items() if key!="wind_m_s"}
    out["wind_m_s"]=rng.uniform(*bounds["wind_m_s"],size=3)
    return out


class DartDynamics:
    """Owns air-force rows for named free darts; use the same step in PPO.

    Wind and randomization are opt-in. Randomization mutates model parameters:
    each worker must own a model. Render/physics-only XML loading has no air loads.
    """
    def __init__(self, model, body_names=("spare_dart_1",), profile=None):
        self.model=model
        self.p=deepcopy(profile if profile is not None else load_profile())
        self.prop=properties(self.p)
        self.ids=[model.body(n).id for n in body_names]
        self.nominal={bid:(model.body_mass[bid],model.body_ipos[bid].copy(),model.body_inertia[bid].copy()) for bid in self.ids}
        self.geoms=np.asarray([g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in self.ids])
        self.nominal_friction=model.geom_friction[self.geoms].copy()
        self.params={"normal_force_scale":1.,"drag_scale":1.,"wind_m_s":np.zeros(3)}
        self.jacp=np.zeros((3,model.nv))
        self.jacr=np.zeros((3,model.nv))

    def randomize(self,data,rng):
        values=sample_parameters(self.p,rng)
        for bid,(mass,com,inertia) in self.nominal.items():
            self.model.body_mass[bid]=mass*values["mass_scale"]
            self.model.body_ipos[bid]=com
            self.model.body_ipos[bid,0]-=values["com_offset_m"]
            self.model.body_inertia[bid]=inertia*values["mass_scale"]*values["inertia_scale"]
        self.model.geom_friction[self.geoms]=self.nominal_friction*values["friction_scale"]
        self.params={"normal_force_scale":values["normal_force_scale"],"drag_scale":values["drag_scale"],"wind_m_s":values["wind_m_s"]}
        mujoco.mj_setConst(self.model,data)
        mujoco.mj_forward(self.model,data)
        return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in values.items()}

    def wrench(self,rotation,velocity_com,omega_world,com_local):
        p=self.p
        arm=rotation @ (np.array([self.prop["aerodynamic_center_x_m"],0,0])-com_local)
        velocity=rotation.T @ (velocity_com + np.cross(omega_world,arm)-self.params["wind_m_s"])
        transverse=velocity[1:]
        transverse_speed=float(np.linalg.norm(transverse))
        speed=float(np.linalg.norm(velocity))
        force_local=np.zeros(3)
        force_local[0]=(-.5*p["air_density_kg_m3"]*p["axial_drag_coefficient"]*
                        math.pi*(p["barrel_diameter_m"]/2)**2*abs(velocity[0])*velocity[0]*self.params["drag_scale"])
        if transverse_speed>1e-12:
            alpha=math.atan2(transverse_speed,abs(velocity[0]))
            # Smooth bounded extension beyond the reference's small-angle regime.
            alpha=p["stall_angle_rad"]*math.tanh(alpha/p["stall_angle_rad"])
            normal=(.5*p["air_density_kg_m3"]*speed**2*p["flight_area_m2"]*
                    p["normal_force_slope_per_rad"]*alpha*self.params["normal_force_scale"])
            normal+=(.5*p["air_density_kg_m3"]*p["crossflow_drag_coefficient"]*
                     p["barrel_diameter_m"]*p["barrel_length_m"]*transverse_speed**2*self.params["drag_scale"])
            force_local[1:]=-normal*transverse/transverse_speed
        force=rotation @ force_local
        omega_local=rotation.T @ omega_world
        roll=(-.5*p["air_density_kg_m3"]*p["flight_area_m2"]*p["flight_span_m"]**3*
              p["roll_drag_coefficient"]*abs(omega_local[0])*omega_local[0])
        torque=np.cross(arm,force)+rotation @ np.array([roll,0,0])
        return force,torque

    def apply(self,data):
        for bid in self.ids:
            mujoco.mj_jacBodyCom(self.model,data,self.jacp,self.jacr,bid)
            force,torque=self.wrench(data.xmat[bid].reshape(3,3),self.jacp @ data.qvel,
                                     self.jacr @ data.qvel,self.model.body_ipos[bid])
            data.xfrc_applied[bid,:3]=force
            data.xfrc_applied[bid,3:]=torque

    def step(self,data):
        # Refresh kinematics before evaluating world-space forces at this state.
        mujoco.mj_forward(self.model,data)
        self.apply(data)
        mujoco.mj_step(self.model,data)
