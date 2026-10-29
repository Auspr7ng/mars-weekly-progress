"""Check live four-finger sensing, per-env independence, transforms and reset."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from sunday_a1_mjlab.flexitac import local_taxels


def main():
    cfg = load_env_cfg("SundayA1-FixedBase-FlexiTac-v0")
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset()
        robot = env.scene["robot"]
        _, action_names = robot.find_joints_by_actuator_names((".*",))
        finger_names = ("l_finger_slide_bottom", "l_finger_slide_top", "r_finger_slide_bottom", "r_finger_slide_top")
        indices = [action_names.index(name) for name in finger_names]
        # Four different states in four parallel worlds, sampled by one kernel.
        targets = torch.tensor([[1., 1., 1., 1.], [-1., -1., -1., -1.],
                                [-1., -1., 1., 1.], [1., 1., -1., -1.]], device="cuda")
        action = torch.zeros(env.action_space.shape, device="cuda")
        action[:, indices] = targets
        # Rotate both moving hand assemblies: taxels/target meshes must follow.
        for joint in ("l_wrist_yaw", "r_wrist_yaw"):
            action[:, action_names.index(joint)] = 1.0
        for _ in range(100):
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
        sensor = env.scene["flexitac"]
        data = sensor.data
        tensor = data.tactile_points_w_per_sensor
        assert tuple(tensor.shape) == (4, 4, 384, 4)
        response = data.response
        assert torch.isfinite(tensor).all() and response.min() >= 0 and response.max() <= 1
        active = (response > 0.01).any(dim=-1).cpu().tolist()
        expected = [[False] * 4, [True] * 4, [True, True, False, False], [False, False, True, True]]
        assert active == expected, (active, expected)
        # Independent forward-kinematic reference for every taxel, every world.
        xpos = env.sim.data.xpos[:, sensor.engine.source_ids].cpu().numpy()
        xmat = env.sim.data.xmat[:, sensor.engine.source_ids].cpu().numpy()
        reference = np.einsum("esij,spj->espi", xmat, local_taxels()) + xpos[:, :, None, :]
        position_error = float(np.max(np.abs(tensor[..., :3].cpu().numpy() - reference)))
        assert position_error < 2e-6, position_error
        # A cached read must not rerun the kernel; a reset invalidates that cache.
        assert sensor.data is data
        summary = {name: data.summary(index) for index, name in enumerate((
            "open", "closed", "left_closed_right_open", "left_open_right_closed"))}
        env.reset()
        after_reset = env.scene["flexitac"].data
        assert (after_reset.response < 0.01).all(), after_reset.summary()
        result = {
            "task": "SundayA1-FixedBase-FlexiTac-v0", "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(), "steps": 100, "num_envs": 4,
            "tactile_shape": [4, 4, 384, 4], "observation_shape": list(obs["actor"].shape),
            "expected_active": expected, "actual_active": active,
            "maximum_taxel_pose_error_m": position_error,
            "reset_clears_closed_response": True, "finite_state": True,
            "phases": summary, "calibrated_pressure": False,
        }
        out = PROJECT.parent / "results/flexitac/mjlab_check.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
