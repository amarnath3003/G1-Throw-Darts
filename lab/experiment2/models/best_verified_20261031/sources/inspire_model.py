"""Convert the official G1 Inspire DFQ hands, including six-actuator couplings."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from grip_demo import ROOT, add, vec

ASSETS = ROOT / "assets/inspire"
PARTS = ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
         "middle_proximal", "ring_proximal", "pinky_proximal")
NAMES = [f"L_{p}_joint" for p in PARTS]
DISTAL = ["L_thumb_distal", "L_index_intermediate", "L_middle_intermediate"]


def origin(element):
    o = element.find("origin")
    if o is None:
        return dict(pos="0 0 0", quat="1 0 0 0")
    q = Rotation.from_euler("xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")).as_quat()
    return dict(pos=o.get("xyz", "0 0 0"), quat=vec(q[[3, 0, 1, 2]]))


def inertial(parent, link):
    source = link.find("inertial")
    i = source.find("inertia")
    tensor = np.array([[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                       [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                       [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]])
    o = origin(source)
    q = np.fromstring(o["quat"], sep=" ")
    r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    tensor = r @ tensor @ r.T
    add(parent, "inertial", pos=o["pos"], mass=source.find("mass").get("value"),
        fullinertia=vec(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))


def build_model(timestep=.0005):
    root = ET.parse(ROOT / "scene.xml").getroot()
    for inc in list(root.findall("include")):
        src = ET.parse(ROOT / inc.get("file")).getroot()
        root.remove(inc)
        root.extend(list(src))
    root.set("model", "G1 five-finger Inspire DFQ - tripod grasp - supported base")
    root.find("compiler").set("meshdir", str(ROOT / "assets/g1/meshes"))
    for tex in root.findall(".//texture[@file]"):
        tex.set("file", str(ROOT / tex.get("file")))
    root.find("option").attrib.update(timestep=str(timestep), iterations="100", tolerance="1e-10", noslip_iterations="20")
    asset, actuator = root.find("asset"), root.find("actuator")
    equality = add(root, "equality")
    g1 = ET.parse(ASSETS / "upstream_g1.urdf").getroot()
    for side in ("right", "left"):
        urdf = ET.parse(ASSETS / f"DFQ_{side}_hand.urdf").getroot()
        links = {l.get("name"): l for l in urdf.findall("link")}
        children = {}
        for j in urdf.findall("joint"):
            children.setdefault(j.find("parent").get("link"), []).append(j)
        wrist_name = f"{side}_wrist_yaw_link"
        wrist = root.find(f".//body[@name='{wrist_name}']")
        for geom in list(wrist.findall("geom")):
            if "rubber_hand" in geom.get("mesh", ""):
                wrist.remove(geom)
        wrist.remove(wrist.find("inertial"))
        inertial(wrist, g1.find(f"link[@name='{wrist_name}']"))

        def attach(parent, parent_name):
            for joint in children.get(parent_name, []):
                name = joint.find("child").get("link")
                link = links[name]
                body = add(parent, "body", name=name, **origin(joint))
                inertial(body, link)
                if joint.get("type") != "fixed":
                    limit = joint.find("limit")
                    jname = joint.get("name")
                    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                    # Coupled thumb distal limits imply a 0..0.5 rad parent range.
                    if "thumb_proximal_pitch" in jname:
                        lower, upper = 0., .5
                    add(body, "joint", name=jname, type="hinge", axis=joint.find("axis").get("xyz"),
                        range=f"{lower} {upper}", damping=".005", armature=".00002",
                        actuatorfrcrange="-1 1")
                    mimic = joint.find("mimic")
                    if mimic is not None:
                        add(equality, "joint", joint1=jname, joint2=mimic.get("joint"),
                            polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0",
                            solref=".002 1", solimp=".99 .999 .001")
                    else:
                        add(actuator, "motor", name=jname, joint=jname, ctrlrange="-1 1", ctrllimited="true")
                for tag in ("visual", "collision"):
                    for k, geom in enumerate(link.findall(tag)):
                        file = Path(geom.find("geometry/mesh").get("filename")).name
                        meshname = file[:-4]
                        if asset.find(f"mesh[@name='{meshname}']") is None:
                            add(asset, "mesh", name=meshname, file=str(ASSETS / "meshes" / file))
                        attrs = dict(name=f"{name}_{tag}_{k}", type="mesh", mesh=meshname, **origin(geom))
                        if tag == "visual":
                            attrs.update(contype="0", conaffinity="0", group="1", density="0", rgba=".78 .80 .82 1")
                        else:
                            attrs.update(group="3", friction=".7 .0001 .00001", condim="4", priority="2",
                                         solref=".003 1", solimp=".95 .99 .001", margin=".00002")
                        add(body, "geom", **attrs)
                attach(body, name)
        attach(wrist, wrist_name)
    pelvis = root.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("joint[@name='floating_base_joint']"))
    world = root.find("worldbody")
    add(world, "geom", name="hand_support", type="capsule", fromto=".07 0 .07 .07 0 .79",
        size=".025", rgba=".85 .57 .18 1", contype="0", conaffinity="0")
    add(world, "geom", name="hand_brace", type="capsule", fromto=".07 0 .79 .30 0 .79",
        size=".018", rgba=".85 .57 .18 1", contype="0", conaffinity="0")
    add(world, "geom", name="hand_drop_mat", type="box", pos=".70 .20 .012",
        size=".45 .4 .012", rgba=".11 .30 .32 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
