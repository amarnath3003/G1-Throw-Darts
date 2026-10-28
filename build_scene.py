"""Build the standalone G1 darts room. Run with the project's MuJoCo Python."""
from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dart_model import add_assets as add_dart_assets, add_dart, load_profile, properties

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BOARD_X = 3.0
BULL_Z = 1.73
OCHE_X = BOARD_X - 2.37
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def font(size: int):
    for name in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def textures():
    image = Image.new("RGB", (2048, 2048), "#171c1e")
    draw = ImageDraw.Draw(image)
    c, scale = 1024, 2048 / .451
    def disk(radius, color):
        r = radius * scale
        draw.ellipse((c-r, c-r, c+r, c+r), fill=color)
    disk(.223, "#343c3e")
    disk(.218, "#101719")
    for i in range(20):
        start, end = -99 + i*18, -81 + i*18
        for radius, color in ((.170, "#c84342" if i % 2 == 0 else "#16735b"),
                              (.162, "#252827" if i % 2 == 0 else "#e4d4b2"),
                              (.107, "#c84342" if i % 2 == 0 else "#16735b"),
                              (.099, "#252827" if i % 2 == 0 else "#e4d4b2")):
            r = radius * scale
            draw.pieslice((c-r, c-r, c+r, c+r), start, end, fill=color)
    for radius in (.170, .162, .107, .099, .0159, .00635):
        r = radius * scale
        draw.ellipse((c-r, c-r, c+r, c+r), outline="#b6aaa0", width=3)
    for i in range(20):
        a = math.radians(i*18-9)
        draw.line((c+.0159*scale*math.sin(a), c-.0159*scale*math.cos(a),
                   c+.170*scale*math.sin(a), c-.170*scale*math.cos(a)), fill="#b6aaa0", width=3)
    disk(.0159, "#16735b")
    disk(.00635, "#c84342")
    for i, number in enumerate(SECTORS):
        a = math.radians(i*18)
        draw.text((c+.195*scale*math.sin(a), c-.195*scale*math.cos(a)), str(number),
                  font=font(92), fill="#e5dfcc", anchor="mm", stroke_width=1)
    image.save(ASSETS / "dartboard.png")
    for name, title, subtitle in (
        ("header", "G1  /  DARTS LAB", "HUMAN MOTION  >  ROBOT LEARNING"),
        ("lane", "THROW LINE", "2.37 m TO BOARD FACE"),
        ("panel", "PRECISION THROUGH PRACTICE", "OBSERVE   /   RETARGET   /   LEARN"),
    ):
        im = Image.new("RGB", (1536, 384), "#12292e")
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 16, 384), fill="#f2a44f")
        d.text((65, 125), title, font=font(76), fill="#f2e7d1", anchor="lm")
        d.text((68, 255), subtitle, font=font(38), fill="#8cb9b8", anchor="lm")
        im.save(ASSETS / f"{name}.png")


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def vector(values):
    return " ".join(f"{x:.9g}" for x in values)


def disk_mesh(asset, name, radius=.2255, count=256):
    vertices = [[0, 0, 0]]
    uv = [[.5, .5]]
    for i in range(count):
        angle = 2*math.pi*i/count
        y, z = radius*math.sin(angle), radius*math.cos(angle)
        vertices.append([0, y, z])
        uv.append([.5-y/(2*radius), .5-z/(2*radius)])
    faces = [[0, i+1, 1+(i+1) % count] for i in range(count)]
    add(asset, "mesh", name=name, inertia="shell", vertex=vector(np.ravel(vertices)),
        face=vector(np.ravel(faces)), texcoord=vector(np.ravel(uv)))


def sign_mesh(asset, name, width, height):
    add(asset, "mesh", name=name, inertia="shell",
        vertex=f"0 {-width/2} {-height/2} 0 {width/2} {-height/2} 0 {width/2} {height/2} 0 {-width/2} {height/2}",
        face="0 2 1 0 3 2", texcoord="1 1 0 1 0 0 1 0")


def build():
    profile = load_profile()
    dart_properties = properties(profile)
    ASSETS.mkdir(parents=True, exist_ok=True)
    robot_dir = ASSETS / "g1"
    (robot_dir / "meshes").mkdir(parents=True, exist_ok=True)
    upstream = ROOT.parent / "scrap test/unitree_mujoco/unitree_robots/g1"
    model_path = robot_dir / "robot.xml"
    if (upstream / "g1_29dof.xml").is_file():
        robot = ET.parse(upstream / "g1_29dof.xml")
        for mesh in robot.findall("./asset/mesh"):
            mesh_path = robot_dir / "meshes" / mesh.attrib["file"]
            if not mesh_path.is_file():
                shutil.copy2(upstream / "meshes" / mesh.attrib["file"], mesh_path)
        # The vendored copy lets this scene rebuild outside the larger workspace.
        robot.find("compiler").set("meshdir", "assets/g1/meshes")
        robot.find("./worldbody/body[@name='pelvis']").set("pos", ".30 0 .793")
        robot.write(model_path, encoding="unicode")
        shutil.copy2(upstream.parent.parent / "LICENSE", robot_dir / "LICENSE")
    elif not model_path.is_file():
        raise FileNotFoundError("Expected the upstream G1 model or the bundled assets/g1/robot.xml")
    textures()
    scene = ET.Element("mujoco", model="G1 Darts Lab")
    add(scene, "include", file="assets/g1/robot.xml")
    add(scene, "option", timestep=".0005", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    add(scene, "statistic", center="1 0 1.1", extent="4")
    visual = add(scene, "visual")
    add(visual, "global", offwidth="1600", offheight="1000", azimuth="135", elevation="-17")
    add(visual, "headlight", ambient=".42 .44 .45", diffuse=".7 .7 .68", specular=".25 .25 .25")
    add(visual, "quality", shadowsize="4096", offsamples="4")
    asset = add(scene, "asset")
    add_dart_assets(asset, profile)
    add(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1=".25 .20 .15", rgb2=".30 .25 .19", width="512", height="512")
    add(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="2 2", texuniform="true", reflectance=".04")
    for name in ("dartboard", "header", "lane", "panel"):
        add(asset, "texture", name=f"{name}_tex", type="2d", file=f"assets/{name}.png")
        add(asset, "material", name=f"{name}_mat", texture=f"{name}_tex", specular=".1", shininess=".1", emission=".12")
    disk_mesh(asset, "board_face")
    for name, width, height in (("header", 1.55, .3875), ("lane", .85, .2125), ("panel", 1.5, .375)):
        sign_mesh(asset, f"{name}_mesh", width, height)
    world = add(scene, "worldbody")
    add(world, "light", name="key", pos=".2 -1.2 3", dir=".3 .1 -1", diffuse=".85 .83 .75", castshadow="true")
    add(world, "light", name="fill", pos="1.7 1.8 2.7", dir="0 -.3 -1", diffuse=".45 .58 .62", castshadow="false")
    add(world, "light", name="board_light", pos="2.65 0 2.45", dir=".4 0 -1", diffuse=".7 .65 .52", castshadow="false")
    add(world, "camera", name="room", pos="-2.65 -4.2 2.65", xyaxes=".64 -.77 0 .22 .18 .96", fovy="49")
    add(world, "camera", name="board", pos="1.65 0 1.73", xyaxes="0 -1 0 0 0 1", fovy="36")
    add(world, "camera", name="robot", pos="1.65 -2.0 1.65", xyaxes=".83 .56 0 -.13 .19 .97", fovy="44")
    add(world, "camera", name="side", pos="1.2 -4.2 1.75", xyaxes="1 0 0 0 0 1", fovy="51")
    add(world, "camera", name="darts", pos=".46 -1.75 1.16", xyaxes="1 0 0 0 .52 .85", fovy="38")
    def box(name, pos, size, rgba, **extra):
        return add(world, "geom", name=name, type="box", pos=pos, size=size, rgba=rgba, **extra)
    add(world, "geom", name="floor", type="plane", size="6 6 .05", material="floor_mat", friction=".9 .01 .001")
    box("front_wall", "3.14 0 1.6", ".06 2.45 1.6", ".09 .17 .19 1")
    box("left_wall", ".65 2.46 1.6", "2.55 .05 1.6", ".15 .22 .23 1")
    # Open camera-side wall and ceiling for an unobstructed inspection view.
    box("right_wall_low", ".65 -2.46 .19", "2.55 .05 .19", ".15 .22 .23 1")
    box("wall_base", "3.069 0 .06", ".012 2.45 .06", ".035 .07 .08 1")
    box("accent_left", "3.065 -1.01 1.65", ".012 .012 1.18", ".96 .52 .19 1", contype="0", conaffinity="0")
    box("accent_right", "3.065 1.01 1.65", ".012 .012 1.18", ".96 .52 .19 1", contype="0", conaffinity="0")
    box("lane_mat", "-.22 0 .004", "1.0 .53 .004", ".055 .12 .135 1", contype="0", conaffinity="0")
    box("oche", f"{OCHE_X} 0 .009", ".012 .61 .009", ".98 .62 .24 1", contype="0", conaffinity="0")
    for y in (-.54, .54):
        box(f"lane_border_{y}", "-.22 "+str(y)+" .009", "1.0 .006 .001", ".38 .59 .58 1", contype="0", conaffinity="0")
    for x in (.95, 1.25, 1.55, 1.85, 2.15, 2.45):
        box(f"lane_dash_{x}", f"{x} 0 .001", ".06 .005 .001", ".55 .47 .33 1", contype="0", conaffinity="0")
    board = add(world, "body", name="dartboard", pos=f"{BOARD_X} 0 {BULL_Z}")
    add(board, "geom", name="surround", type="cylinder", pos=".04 0 0", quat=".707106781 0 .707106781 0", size=".33 .018", rgba=".038 .055 .06 1")
    add(board, "geom", name="board_collision", type="cylinder", pos=".02 0 0", quat=".707106781 0 .707106781 0", size=".2255 .02", rgba=".12 .14 .14 1")
    add(board, "geom", name="board_art", type="mesh", mesh="board_face", pos="-.0004 0 0", material="dartboard_mat", contype="0", conaffinity="0", mass="0")
    add(board, "site", name="bullseye", pos="-.0005 0 0", size=".00635", rgba="1 0 0 0")
    add(board, "site", name="board_frame", pos="0 0 0", size=".005", rgba="0 1 0 0")
    add(world, "site", name="oche_center", pos=f"{OCHE_X} 0 0", size=".005", rgba="0 1 0 0")
    for name, pos in (("header", "3.06 0 2.51"), ("panel", "3.06 0 .67")):
        add(world, "geom", name=f"{name}_sign", type="mesh", mesh=f"{name}_mesh", pos=pos, material=f"{name}_mat", contype="0", conaffinity="0", mass="0")
    # Side equipment station; well clear of the throwing lane.
    box("table_top", ".48 -1.20 .77", ".48 .28 .025", ".24 .29 .28 1")
    for x in (.08, .88):
        for y in (-1.42, -.98):
            box(f"table_leg_{x}_{y}", f"{x} {y} .375", ".022 .022 .375", ".07 .12 .13 1")
    box("dart_tray", ".45 -1.19 .801", ".20 .15 .006", ".06 .10 .11 1")
    for i, y in enumerate((-1.28, -1.19, -1.10)):
        add_dart(world, f"spare_dart_{i+1}", f".46 {y} .832", profile, free=(i == 0))
    # Wall-mounted sample dart makes the scale and flight silhouette easy to inspect.
    add_dart(world, "display_dart", "2.985 .080 1.755", profile)
    # A subtle equipment shelf below the board.
    box("board_shelf", "2.94 0 1.18", ".13 .43 .018", ".10 .14 .14 1")
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(ROOT / "scene.xml", encoding="unicode")
    settings = {
        "board_face_x_m": BOARD_X, "bull_height_m": BULL_Z, "oche_x_m": OCHE_X,
        "oche_to_board_m": 2.37, "board_outer_radius_m": .2255,
        "scoring_outer_radius_m": .170, "inner_bull_radius_m": .00635,
        "outer_bull_radius_m": .0159, "dart_mass_kg": profile["mass_kg"],
        "dart_properties": dart_properties,
        "dart_profile": "dart_profile.json",
        "dart_status": "Experimental reference mass and COM; literature-inferred transverse inertia, approximate geometry and uncalibrated contact/drag. Air loads require DartDynamics.step(). No verified sim-to-real transfer or sticking model.",
        "robot_status": "Existing 29-DoF G1 with rigid hands; inspection pose, no trained balance controller.",
        "sources": {"board_setup": "https://www.pdc-europe.tv/wiki/rules/", "dart_reference": profile["reference"]},
    }
    (ROOT / "scene_config.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    # Export the same trainable physical asset in isolation, for controller and flight tests.
    standalone = ET.Element("mujoco", model="reference_dart")
    add(standalone, "option", timestep=".0005", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    dart_asset = add(standalone, "asset")
    add_dart_assets(dart_asset, profile)
    dart_world = add(standalone, "worldbody")
    add_dart(dart_world, "dart", "0 0 1.5", profile, free=True)
    ET.indent(standalone, space="  ")
    ET.ElementTree(standalone).write(ASSETS / "dart.xml", encoding="unicode")
    print(f"Built {ROOT / 'scene.xml'}")


if __name__ == "__main__":
    build()
