"""Run four FlexiTac arrays on the latest full Sunday A1 model."""
import argparse
import json
import time
import webbrowser
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

from sunday_a1_mjlab.flexitac import FourFingerFlexiTac, attach_patch_sites
from sunday_a1_mjlab.tactile_display import TactileViewer, save_snapshot

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "src/sunday_a1_mjlab/assets"
OUTPUT = PROJECT.parent / "results/flexitac"
JOINTS = ("l_finger_slide_bottom", "l_finger_slide_top", "r_finger_slide_bottom", "r_finger_slide_top")


def make_model():
    spec = mujoco.MjSpec.from_file(str(ASSETS / "scene.xml"))
    attach_patch_sites(spec)
    model = spec.compile()
    # Persist the placement guides. Active Warp sampling lives in Python.
    spec.to_file(str(ASSETS / "scene_flexitac.xml"))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("mjlab", "mujoco"), default="mjlab")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--duration", type=float, default=0, help="Optional wall-clock limit for viewer checks")
    args = parser.parse_args()
    wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    backend = None
    if args.backend == "mjlab":
        from sunday_a1_mjlab.tactile_backend import MjlabTactileBackend
        backend = MjlabTactileBackend(args.device)
        model, data = backend.model, backend.data
    else:
        model = make_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sensor = FourFingerFlexiTac(model, device=args.device)
        actuator_ids = [model.actuator(n).id for n in JOINTS]
    frames, timestamps = [], []

    def step(targets, nstep=10):
        if backend is not None:
            return backend.step(targets)
        data.ctrl[actuator_ids] = targets
        mujoco.mj_step(model, data, nstep=nstep)
        # Refresh poses after integration, as mjlab does before observations.
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Non-finite MuJoCo state")
        return sensor.sample_mujoco(data)

    if args.headless:
        report = {}
        for name, targets, expected in (
            ("open", [0.03] * 4, [False] * 4),
            ("closed", [0.] * 4, [True] * 4),
            ("left_closed_right_open", [0., 0., 0.03, 0.03], [True, True, False, False]),
            ("left_open_right_closed", [0.03, 0.03, 0., 0.], [False, False, True, True]),
        ):
            for _ in range(100):
                result = step(targets)
                frames.append(result.tactile_points_w_per_sensor[0].detach().cpu().numpy().copy())
                timestamps.append(data.time)
            actual = backend.actual_positions() if backend else np.array([data.joint(n).qpos[0] for n in JOINTS])
            assert np.max(np.abs(actual - targets)) < 0.001, (name, actual)
            response = result.response.detach().cpu().numpy()[0]
            assert np.isfinite(response).all() and response.min() >= 0 and response.max() <= 1
            active = (response > 0.01).any(axis=1).tolist()
            assert active == expected, (name, active, expected)
            save_snapshot(result, OUTPUT, name)
            report[name] = {"joints_m": actual.tolist(), "sensors": result.summary()}
        np.savez_compressed(OUTPUT / "open_close_readings.npz", time_s=np.array(timestamps),
                            tactile_points_w_per_sensor=np.array(frames))
        report["contract"] = {"backend": args.backend, "shape": [1, 4, 384, 4], "rows": 12, "cols": 32,
                              "pitch_m": 0.002, "device": args.device,
                              "response": "normalized unsigned mesh-distance shell; not calibrated pressure"}
        (OUTPUT / f"{args.backend}_demo_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if backend:
            backend.close()
        return

    result = step([0.03] * 4)
    viewer = TactileViewer(model, data, port=args.port, prefix=backend.prefix if backend else "", backend=args.backend)
    url = f"http://127.0.0.1:{viewer.server.get_port()}"
    print(f"Four FlexiTac arrays ready: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    started, last_draw, count = time.perf_counter(), 0.0, 0
    try:
        while not viewer.stopped and (not args.duration or time.perf_counter() - started < args.duration):
            tick = time.perf_counter()
            if not viewer.paused.value:
                if viewer.auto.value:
                    target = 0.03 if int(data.time / 3) % 2 == 0 else 0.0
                    targets = [target] * 4
                else:
                    targets = [viewer.left.value / 1000] * 2 + [viewer.right.value / 1000] * 2
                result = step(targets)
                count += 1
            if tick - last_draw >= 0.1:
                viewer.update(result, data.time, count)
                last_draw = tick
            if viewer.want_snapshot:
                save_snapshot(result, OUTPUT, "live_heatmaps")
                viewer.want_snapshot = False
                print(f"Saved current heatmaps and readings to {OUTPUT}", flush=True)
            time.sleep(max(0, 0.02 - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        if backend:
            backend.close()


if __name__ == "__main__":
    main()
