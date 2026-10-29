"""Run a bounded GPU rollout and write measured setup status."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import warp as wp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--task", default="SundayA1-FixedBase-JointCheck-v0")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch cannot see a GPU")
    project = Path(__file__).resolve().parents[1]
    wp.config.kernel_cache_dir = str(project / ".cache/warp")
    os.environ.setdefault("MPLCONFIGDIR", str(project / ".cache/matplotlib"))
    import mjlab.tasks  # Registers official tasks and installed plugin.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    if args.task not in list_tasks():
        raise RuntimeError(f"Task plugin not loaded: {args.task}")
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    report = {}
    try:
        obs, _ = env.reset()
        action = torch.zeros(env.action_space.shape, device=args.device)
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite reward")
            for name, value in obs.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite observation: {name}")
        robot = env.scene["robot"]
        if not torch.isfinite(robot.data.joint_pos).all():
            raise RuntimeError("Non-finite joint state")
        root_positions = robot.data.root_link_pos_w.detach().cpu()
        if args.num_envs > 1 and torch.unique(root_positions, dim=0).shape[0] != args.num_envs:
            raise RuntimeError("Fixed-base robots were not reset to separate env origins")
        report = {
            "packages": {n: importlib.metadata.version(n) for n in (
                "mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch")},
            "task": args.task, "device": args.device,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "warp_cuda_available": wp.is_cuda_available(),
            "num_envs": args.num_envs, "steps": args.steps,
            "action_shape": list(action.shape),
            "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
            "root_positions": root_positions.tolist(),
            "finite_state": True, "tactile_migrated": "flexitac" in env.scene.sensors,
        }
    finally:
        env.close()
    output = Path(__file__).resolve().parents[2] / "results"
    output.mkdir(exist_ok=True)
    filename = "flexitac_mjlab_check.json" if report["tactile_migrated"] else "mjlab_check.json"
    (output / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
