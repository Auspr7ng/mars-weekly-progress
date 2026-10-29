"""Native MuJoCo preview; does not require Isaac Sim or mjlab."""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "src/sunday_a1_mjlab/assets"
FINGERS = ("l_finger_slide_bottom", "l_finger_slide_top", "r_finger_slide_bottom", "r_finger_slide_top")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--manual", action="store_true", help="L/R independently toggle left/right grippers")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "scene.xml"))
    data = mujoco.MjData(model)
    ids = [model.actuator(name).id for name in FINGERS]

    def step(target):
        data.ctrl[ids] = target
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError("Non-finite robot state")

    if args.headless:
        positions = {}
        for name, target in (
            ("open", [0.03] * 4), ("closed", [0.0] * 4),
            ("left_open_right_closed", [0.03, 0.03, 0.0, 0.0]),
            ("left_closed_right_open", [0.0, 0.0, 0.03, 0.03]),
        ):
            for _ in range(args.steps):
                step(target)
            actual = [float(data.joint(j).qpos[0]) for j in FINGERS]
            positions[name] = actual
            if max(abs(value - requested) for value, requested in zip(actual, target)) > 0.001:
                raise RuntimeError(f"Finger target {target} was not reached: {actual}")
        report = {"mujoco": mujoco.__version__, "nq": model.nq,
                  "nu": model.nu, "finger_positions_m": positions,
                  "finite_state": True}
        output = ASSETS.parents[3] / "results"
        output.mkdir(exist_ok=True)
        (output / "preview_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        from mujoco import viewer as mjviewer
        targets = [0.03] * 4
        manual = [args.manual]

        def on_key(key):
            if key in (ord("L"), ord("R")):
                manual[0] = True
                start = 0 if key == ord("L") else 2
                target = 0.0 if targets[start] > 0.015 else 0.03
                targets[start:start + 2] = [target, target]

        print("L: toggle left gripper; R: toggle right gripper. Each hand moves independently.")
        with mjviewer.launch_passive(model, data, key_callback=on_key) as viewer:
            viewer.cam.distance = 2.2
            info = json.loads((ASSETS / "model_info.json").read_text())
            viewer.cam.lookat[:] = (0, 0, -info["base_height"] / 2)
            while viewer.is_running():
                start = time.perf_counter()
                step(targets if manual[0] else (0.03 if int(data.time / 3) % 2 == 0 else 0.0))
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
