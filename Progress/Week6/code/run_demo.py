"""Native Xbox input -> Week4 leg policy + bilateral arm/gripper teleoperation in MuJoCo."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import webbrowser

WEEK = Path(__file__).resolve().parents[1]
BASE = WEEK.parent / "Week4"
for name, subdir in {
    "MPLCONFIGDIR": "matplotlib",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "XDG_CACHE_HOME": "xdg",
}.items():
    os.environ.setdefault(name, str(WEEK / ".cache" / subdir))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
sys.path[1:1] = [
    str(WEEK / ".deps"),
    str(WEEK.parent / "Week5/.deps"),
    str(BASE / "code/vendor"),
    str(BASE / "code/src"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--hold-lb",
        action="store_true",
        help="Require LB while moving (default: direct spring-centred stick control)",
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error("Invalid port/duration")
    (WEEK / "results").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "config/accepted_policy.json").read_text(encoding="utf-8"))
    checkpoint = BASE / manifest["checkpoint"]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if not manifest.get("accepted") or digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Week4 accepted checkpoint hash mismatch")
    spec = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
    model_path = checkpoint.parent / "dynamics/sunday_a1_locomotion.xml"
    nominal = checkpoint.parent / "dynamics/arm_home.json"
    if digest(model_path) != spec["model_sha256"] or digest(nominal) != spec["nominal_sha256"]:
        raise ValueError("Week4 archived model/nominal hash mismatch")
    from controller import Controller
    from native_input import NativeInput
    from week4_mapping import map_left_stick

    control = Controller(
        limits=manifest["joystick_limits"], require_lb=args.hold_lb, mapper=map_left_stick
    )
    reader = NativeInput(control)
    env = ui = None
    try:
        reader.start()
        import warp as wp

        # Keep generated kernels and input/output files in Week6.
        wp.config.kernel_cache_dir = str(WEEK / ".cache/warp")
        import mujoco
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from sunday_locomotion.env_cfg import make_env, agent_cfg
        from simulation_player import run_native_web
        from upper_body import ArmTeleop
        from week3_tactile.flexitac import FourFingerFlexiTac
        from week3_tactile.feet_tactile import FootTactileEngine
        from viewer import Viewer

        # Scripted diagnostics do not bind a viewer port and can run beside it.
        if not getattr(Viewer, "headless", False):
            with socket.socket() as connection:
                connection.settimeout(0.3)
                if connection.connect_ex(("127.0.0.1", args.port)) == 0:
                    raise RuntimeError(
                        f"Port {args.port} already in use. Reopen http://127.0.0.1:{args.port}/ or choose --port."
                    )
        torch.set_num_threads(4)
        revision = spec["revision"]
        cfg = make_env(
            1,
            spec["stage"],
            spec["whole_body"],
            play=True,
            gait_phase=spec["gait_phase"],
            revision=revision,
            nominal_path=nominal,
            model_path=model_path,
        )
        cfg.rewards = {}
        cfg.episode_length_s = 3600.0
        cfg.auto_reset = False
        cfg.terminations.pop("boundary", None)
        cfg.terminations.pop("time_out", None)
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = MjlabOnPolicyRunner(
            wrapper, asdict(agent_cfg(revision)), log_dir=None, device="cuda:0"
        )
        runner.load(str(checkpoint))
        env.command_manager.get_term("twist").override = [0.0, 0.0, 0.0]
        env.reset(seed=741)
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        hands = FourFingerFlexiTac(model, prefix="robot/")
        feet = FootTactileEngine(model, prefix="robot/")
        upper_body = ArmTeleop(model, data, json.loads(nominal.read_text())["joint_pos"])
        ui = Viewer(model, data, control, reader, args.port, upper_body)
        url = f"http://127.0.0.1:{args.port}/"
        headless = getattr(ui, "headless", False)
        info = dict(
            pid=os.getpid(),
            url=None if headless else url,
            port=None if headless else args.port,
            input_backend=getattr(reader, "backend", "pygame_sdl_standard"),
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            model_sha256=digest(model_path),
            limits=control.limits,
            physics_backend="native_mujoco",
            actor_device="cpu",
            controller=control.sample(),
            walking_input="left_stick_xy",
            command_mapper="Week4 joystick_to_command",
            right_stick="stationary_bilateral_arm_lift_spread",
            walking_at_any_arm_pose=True,
            buttons="A open / Y close both hands / X optional Home / Menu reset",
            upper_body=upper_body.status(),
            acceptance="experimental simulation only",
        )
        info_name = "scripted_server_info.json" if headless else "server_info.json"
        (WEEK / "results" / info_name).write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
        print("WEEK6_SCRIPTED_READY" if headless else "WEEK6_READY " + url, flush=True)
        if not args.no_browser and not headless:
            webbrowser.open(url)
        run_native_web(
            model,
            data,
            env,
            runner.get_inference_policy(device="cpu"),
            ui,
            hands,
            feet,
            "Week6 experimental arms + R25-300 legs",
            args.duration,
            upper_body=upper_body,
        )
    finally:
        reader.close()
        if ui:
            ui.close()
        if env:
            env.close()


if __name__ == "__main__":
    main()
