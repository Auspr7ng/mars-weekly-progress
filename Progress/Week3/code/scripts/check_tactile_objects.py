"""Probe all four arrays with unnamed primitive/mesh objects and changing poses."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

from sunday_a1_mjlab.flexitac import FINGERS, FourFingerFlexiTac, local_taxels
from sunday_a1_mjlab.tasks import ASSETS


def main():
    spec = mujoco.MjSpec.from_file(str(ASSETS / "scene.xml"))
    probes = {
        "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, (.014, 0, 0)),
        "box": (mujoco.mjtGeom.mjGEOM_BOX, (.010, .014, .020)),
        "capsule": (mujoco.mjtGeom.mjGEOM_CAPSULE, (.014, .020, 0)),
        "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, (.014, .020, 0)),
        "ellipsoid": (mujoco.mjtGeom.mjGEOM_ELLIPSOID, (.010, .014, .020)),
        "mesh": (mujoco.mjtGeom.mjGEOM_MESH, (.010, .014, .020)),
    }
    # An external triangle object with no target name registration.
    mesh = spec.add_mesh(name="external_cube")
    mesh.uservert = np.array([(x, y, z) for x in (-.010, .010)
                            for y in (-.014, .014) for z in (-.020, .020)]).ravel().tolist()
    mesh.userface = [0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6,
                     0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3,
                     0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5]
    for i, (name, (kind, size)) in enumerate(probes.items()):
        body = spec.worldbody.add_body(name="object_" + name, pos=(2 + i, 0, 1))
        body.add_freejoint(name="joint_" + name)
        geom = body.add_geom(name="external_" + name, type=kind, size=size, mass=.04)
        if name == "mesh":
            geom.meshname = "external_cube"
    model = spec.compile()
    data = mujoco.MjData(model)
    for name in ("l_finger_slide_bottom", "l_finger_slide_top", "r_finger_slide_bottom", "r_finger_slide_top"):
        data.joint(name).qpos[0] = .03
    mujoco.mj_forward(model, data)
    engine = FourFingerFlexiTac(model)
    assert engine.sample_mujoco(data).response.max().item() == 0
    report = {}
    taxel = 6 * 32 + 16
    for name in probes:
        values = []
        for index, finger in enumerate(FINGERS):
            bid = model.body(finger.link).id
            rotation = data.xmat[bid].reshape(3, 3)
            point = data.xpos[bid] + rotation @ local_taxels()[index, taxel]
            normal = rotation @ np.array((0., finger.inward, 0.))
            q = data.joint("joint_" + name).qpos
            q[:3] = point + normal * (.014 + .0012)
            q[3:] = (1, 0, 0, 0)
            mujoco.mj_forward(model, data)
            result = engine.sample_mujoco(data)
            value = result.response[0, index, taxel].item()
            distance = result.distance_m[0, index, taxel].item()
            assert value > .1, (name, finger.name, value, distance)
            assert abs(distance - .0012) < .0001, (name, finger.name, distance)
            values.append({"finger": finger.name, "response": value, "distance_m": distance})
            # No grasp/reset callback: moving the object away alone clears data.
            q[:3] = (4, 0, 1)
            mujoco.mj_forward(model, data)
            assert engine.sample_mujoco(data).response.max().item() == 0, name
        report[name] = values
    report["passed"] = True
    output = PROJECT.parent / "results/ball_grasp/object_sensing_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
