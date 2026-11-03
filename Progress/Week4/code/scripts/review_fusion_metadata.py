"""Compare native Fusion observations with the frozen URDF and MJCF."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

WEEK = Path(__file__).resolve().parents[2]
DATA = WEEK / "research/hardware_metadata/2026-11-01"
SOURCE = DATA / "fusion_native_audit_20261101_200848.json"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def com_inertia(props):
    assert props["getXYZMomentsOfInertia"][0]
    xx, yy, zz, xy, yz, xz = props["getXYZMomentsOfInertia"][1:]
    inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    com = np.array(props["center_of_mass_cm"])
    inertia -= props["mass_kg"] * (com @ com * np.eye(3) - np.outer(com, com))
    principal = np.sort(props["getPrincipalMomentsOfInertia"][1:])
    assert np.max(abs(np.sort(np.linalg.eigvalsh(inertia)) - principal)) < 1e-5
    return inertia * 1e-4


def main():
    native = json.loads(SOURCE.read_text(encoding="utf-8"))
    urdf = ET.parse(WEEK / "model/source/sunday_a1.urdf").getroot()
    mjcf = ET.parse(WEEK / "model/sunday_a1_locomotion.xml").getroot()
    mj_joints = {j.get("name"): j for j in mjcf.findall(".//joint") if j.get("name")}
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = {j.get("name"): j for j in urdf.findall("joint") if j.get("type") != "fixed"}
    positions = {"base_link": np.zeros(3)}
    pending = list(urdf.findall("joint"))
    while pending:
        next_pending = []
        for joint in pending:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in positions:
                next_pending.append(joint)
                continue
            origin = joint.find("origin")
            assert np.all(np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") == 0), "Extend FK for rotated link frames."
            positions[child] = positions[parent] + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        assert len(next_pending) < len(pending), "Disconnected URDF graph."
        pending = next_pending

    active = [o for o in native["occurrences"] if o["isReferencedComponent"]]
    by_name = {o["name"]: o for o in active}
    assert len(active) == len(by_name) == 28
    assert len(native["joints"]) == len(joints) == 27
    records, csv_rows = [], []
    for joint in native["joints"]:
        name = joint["name"]
        source = joints[name]
        motion = joint["motion"]
        linear = "slideLimits" in motion
        limits = motion["slideLimits" if linear else "rotationLimits"]
        axis = np.array(motion["slideDirectionVector" if linear else "rotationAxisVector"])
        axis /= np.linalg.norm(axis)
        source_axis = np.fromstring(source.find("axis").get("xyz"), sep=" ")
        source_axis /= np.linalg.norm(source_axis)
        child = source.find("child").get("link")
        parent = source.find("parent").get("link")
        cad_child = joint["occurrenceOne"]["name"].replace(":", "_")
        cad_parent = joint["occurrenceTwo"]["name"].replace(":", "_")
        topology_ok = child == cad_child and (parent == cad_parent or (parent == "base_link" and cad_parent == "base_link_1"))
        assert topology_ok
        cad_body = by_name[joint["occurrenceOne"]["name"]]
        inertial_com = np.fromstring(links[child].find("inertial/origin").get("xyz"), sep=" ")
        zero_com_delta = float(np.linalg.norm(positions[child] + inertial_com - cad_body["physical_properties"]["center_of_mass_m"]))
        bounded = limits["isMinimumValueEnabled"] and limits["isMaximumValueEnabled"]
        bounds = [limits["minimumValue_SI"], limits["maximumValue_SI"]] if bounded else None
        old_limit = source.find("limit")
        urdf_error = max(abs(bounds[0] - float(old_limit.get("lower"))), abs(bounds[1] - float(old_limit.get("upper")))) if bounded else None
        mj_range = np.fromstring(mj_joints[name].get("range", ""), sep=" ")
        mj_error = float(np.max(abs(mj_range - bounds))) if bounded else None
        record = {
            "name": name, "type": "prismatic" if linear else "revolute", "units": "m" if linear else "rad",
            "effective_bounds": bounds, "minimum_enabled": limits["isMinimumValueEnabled"],
            "maximum_enabled": limits["isMaximumValueEnabled"], "axis_in_cad_world_at_zero": axis.tolist(),
            "urdf_axis_error": float(np.linalg.norm(axis - source_axis)),
            "cad_saved_q": motion["slideValue" if linear else "rotationValue"],
            "zero_pose_comparison_error_m": zero_com_delta, "topology_matches_urdf": topology_ok,
            "rotary_pivot_error_m": None if linear else float(np.linalg.norm(np.array(joint["geometryOneTransform"]["translation_m"]) - positions[child])),
            "urdf_bound_error_SI": urdf_error, "mjcf_bound_error_SI": mj_error,
            "rest_enabled": limits["isRestValueEnabled"], "rest_value_SI": limits["restValue_SI"],
            "hardware_encoder_zero": None, "hardware_servo_direction": None,
        }
        records.append(record)
        csv_rows.append({"joint": name, "type": record["type"], "unit": record["units"],
            "min_enabled": record["minimum_enabled"], "max_enabled": record["maximum_enabled"],
            "lower": bounds[0] if bounds else "", "upper": bounds[1] if bounds else "",
            "axis_x_CAD_world": axis[0], "axis_y_CAD_world": axis[1], "axis_z_CAD_world": axis[2],
            "CAD_saved_q": record["cad_saved_q"], "URDF_axis_error": record["urdf_axis_error"],
            "zero_geometry_COM_error_m": zero_com_delta, "parent": parent, "child": child})

    mass_rows = []
    for occurrence in active:
        name = occurrence["name"]
        urdf_name = "base_link" if name == "base_link:1" else name.replace(":", "_")
        props = occurrence["physical_properties"]
        cad_inertia = com_inertia(props)
        materials = sorted({body["material"]["name"] for body in occurrence["body_materials"]})
        mass_rows.append({"occurrence": name, "urdf_link": urdf_name,
            "CAD_material_mass_kg": props["mass_kg"], "material": ";".join(materials),
            "body_count": occurrence["body_count"], "CAD_COM_x_m": props["center_of_mass_m"][0],
            "CAD_COM_y_m": props["center_of_mass_m"][1], "CAD_COM_z_m": props["center_of_mass_m"][2],
            "CAD_COM_Ixx_kgm2": cad_inertia[0, 0], "CAD_COM_Iyy_kgm2": cad_inertia[1, 1],
            "CAD_COM_Izz_kgm2": cad_inertia[2, 2], "CAD_COM_Ixy_kgm2": cad_inertia[0, 1],
            "CAD_COM_Ixz_kgm2": cad_inertia[0, 2], "CAD_COM_Iyz_kgm2": cad_inertia[1, 2],
            "mass_status": "steel-material calculation; not measured"})

    root = by_name["base_link:1"]
    result = {
        "source_audit": str(SOURCE), "source_audit_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_archive_sha256": native["reference_archive_sha256"], "source_context": native["local_import_context"],
        "native_document_modified_before": native["document_modified_before"],
        "native_document_modified_after": native["document_modified_after"],
        "native_joints": len(records), "bounded_joints": sum(r["effective_bounds"] is not None for r in records),
        "max_urdf_axis_error": max(r["urdf_axis_error"] for r in records),
        "max_urdf_bound_error": max(r["urdf_bound_error_SI"] or 0 for r in records),
        "max_mjcf_bound_error": max(r["mjcf_bound_error_SI"] or 0 for r in records),
        "max_rotary_pivot_error_m": max(r["rotary_pivot_error_m"] or 0 for r in records),
        "max_zero_pose_com_error_m": max(r["zero_pose_comparison_error_m"] for r in records),
        "all_CAD_saved_q_zero": all(r["cad_saved_q"] == 0 for r in records),
        "all_topology_matches": all(r["topology_matches_urdf"] for r in records),
        "all_occurrence_count": len(native["occurrences"]), "joint_connected_reference_occurrences": len(active),
        "additional_nonreference_geometry_occurrences": sum(not o["isReferencedComponent"] and o["body_count"] > 0 for o in native["occurrences"]),
        "additional_empty_occurrences": sum(o["body_count"] == 0 for o in native["occurrences"]),
        "CAD_total_mass_including_duplicates_kg": native["root_physical_properties"]["mass_kg"],
        "CAD_joint_connected_mass_kg": sum(o["physical_properties"]["mass_kg"] for o in active),
        "CAD_base_material_mass_kg": root["physical_properties"]["mass_kg"],
        "CAD_base_COM_in_URDF_reference_m": root["physical_properties"]["center_of_mass_m"],
        "CAD_base_COM_inertia_kgm2": com_inertia(root["physical_properties"]).tolist(),
        "CAD_base_uniform_mass_0p7kg_inertia_ESTIMATE_kgm2": (com_inertia(root["physical_properties"]) * .7 / root["physical_properties"]["mass_kg"]).tolist(),
        "mass_credibility": "All 57 BRep bodies of the 28 joint-connected components use steel at 7850 kg/m3; 28 additional geometry copies double the assembly mass. Do not apply these masses to the real robot.",
        "hardware_encoder_zero_and_direction_verified": False, "model_updated": False, "training_resumed": False,
        "model_sha256": hashlib.sha256((WEEK / "model/sunday_a1_locomotion.xml").read_bytes()).hexdigest(),
        "joints": records,
    }
    assert result["bounded_joints"] == 26 and result["all_topology_matches"] and result["all_CAD_saved_q_zero"]
    assert result["max_urdf_axis_error"] < 1e-6 and result["max_urdf_bound_error"] < 1e-6
    assert result["max_rotary_pivot_error_m"] < 1e-6 and result["max_zero_pose_com_error_m"] < 1e-6
    (DATA / "fusion_verified_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(DATA / "fusion_joint_limits_axes_zero.csv", csv_rows)
    write_csv(DATA / "fusion_link_mass_properties.csv", mass_rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ["joints", "source_context"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
