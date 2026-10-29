"""Convert the existing portable URDF without editing the Isaac project."""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "src" / "sunday_a1_mjlab" / "assets"
FINGERS = (
    "l_finger_slide_bottom", "l_finger_slide_top",
    "r_finger_slide_bottom", "r_finger_slide_top",
)


def convert(source: Path) -> dict:
    source = source.resolve(strict=True)
    root = ET.parse(source).getroot()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "meshes").mkdir(exist_ok=True)
    mesh_files = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            raise ValueError("Use the portable URDF with resolved relative mesh paths")
        original = (source.parent / filename).resolve(strict=True)
        shutil.copy2(original, ASSETS / "meshes" / original.name)
        mesh.set("filename", f"meshes/{original.name}")
        mesh_files.add(original.name)

    # Preserve supplied inertias. The massless base stays fixed to the world.
    compiler_parent = root.find("mujoco")
    if compiler_parent is None:
        compiler_parent = ET.SubElement(root, "mujoco")
    ET.SubElement(compiler_parent, "compiler", {
        "meshdir": "meshes", "strippath": "true",
        "discardvisual": "false", "fusestatic": "false",
    })
    intermediate = ASSETS / "sunday_a1_import.urdf"
    ET.ElementTree(root).write(intermediate, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(intermediate))
    converted = ASSETS / "sunday_a1.xml"
    mujoco.mj_saveLastXML(str(converted), model)
    xml = ET.parse(converted)
    robot = xml.getroot()
    # mjlab owns scene physics; only the standalone scene below sets options.
    for option in robot.findall("option"):
        robot.remove(option)
    world = robot.find("worldbody")
    assert world is not None
    # URDF root geometry can otherwise become part of the world. Wrap all top
    # level elements so mjlab can move the entire fixed-base entity with mocap.
    if world.find("body[@name='base_link']") is None:
        base = ET.Element("body", {"name": "base_link"})
        for child in list(world):
            world.remove(child)
            base.append(child)
        world.append(base)
    for geom in robot.findall(".//geom"):
        if geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0":
            continue  # Visual meshes.
        # Basic import check excludes unvalidated convex-mesh self collisions.
        geom.set("contype", "1")
        geom.set("conaffinity", "2")
        geom.set("friction", "0.8 0.005 0.0001")
    ET.indent(robot)
    xml.write(converted, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(converted))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Infer a support height from collision mesh geometry in the zero pose.
    minimum_z = float("inf")
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0:
            continue
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start + count]
        world_vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        minimum_z = min(minimum_z, float(world_vertices[:, 2].min()))
    base_height = max(0.0, -minimum_z + 0.005)
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    if not set(FINGERS).issubset(joint_names):
        raise ValueError("The source is missing one or more of the four finger joints")
    report = {
        "source_urdf": str(source), "mesh_files": len(mesh_files),
        "joint_names": joint_names, "nq": model.nq, "nv": model.nv,
        "nbody": model.nbody, "ngeom": model.ngeom,
        "fixed_base": True, "base_height": base_height,
        "source_base_mass": float(root.find("link[@name='base_link']/inertial/mass").get("value")),
        "self_collision": False, "base_model_has_active_tactile_sensor": False,
    }
    (ASSETS / "model_info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    scene = ET.Element("mujoco", {"model": "sunday_a1_fixed_base_check"})
    ET.SubElement(scene, "include", {"file": "sunday_a1.xml"})
    ET.SubElement(scene, "option", {"timestep": "0.002", "integrator": "implicitfast"})
    scene_world = ET.SubElement(scene, "worldbody")
    ET.SubElement(scene_world, "light", {"pos": "0 -2 3", "dir": "0 1 -1"})
    ET.SubElement(scene_world, "geom", {
        "name": "floor", "type": "plane", "size": "3 3 0.1",
        "pos": f"0 0 {-base_height}", "rgba": "0.22 0.25 0.3 1",
        "contype": "2", "conaffinity": "1",
    })
    actuator = ET.SubElement(scene, "actuator")
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in joint_names:
            continue
        finger = name in FINGERS
        attrs = {
            "name": name, "joint": name,
            "kp": "1000" if finger else "100",
            "kv": "30" if finger else "10",
            "forcerange": "-100 100",
        }
        if joint.get("type") != "continuous":
            limit = joint.find("limit")
            attrs["ctrlrange"] = f"{limit.get('lower')} {limit.get('upper')}"
        ET.SubElement(actuator, "position", attrs)
    ET.indent(scene)
    ET.ElementTree(scene).write(ASSETS / "scene.xml", encoding="utf-8", xml_declaration=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=PROJECT / "model/sunday_a1.urdf")
    convert(parser.parse_args().urdf)
