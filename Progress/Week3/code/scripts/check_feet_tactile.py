"""GPU checks: four 32x32 sole grids, transforms, contact, tilt and reset."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from sunday_a1_mjlab.ball_grasp import TASK_ID
from sunday_a1_mjlab.feet_tactile import FOOT_PATCHES, FOOT_PITCH_M, foot_local_taxels
from sunday_a1_mjlab.tactile_display import save_foot_snapshot


@wp.kernel(enable_backward=False)
def surface_distances(points: wp.array2d(dtype=wp.vec3), mesh_ids: wp.array(dtype=wp.uint64),
                      distances: wp.array2d(dtype=float)):
    patch, point = wp.tid()
    p = points[patch, point]
    hit = wp.mesh_query_point(mesh_ids[patch], p, .1)
    distance = float(.1)
    if hit.result:
        distance = wp.length(p - wp.mesh_eval_position(mesh_ids[patch], hit.face, hit.u, hit.v))
    distances[patch, point] = distance


def main():
    output = PROJECT.parent / "results/feet_tactile"
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 4
    cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset()
        model = env.sim.mj_model
        robot = env.scene["robot"]
        _, names = robot.find_joints_by_actuator_names((".*",))
        action = torch.zeros(env.action_space.shape, device=env.device)
        for name in ("l_finger_slide_bottom", "l_finger_slide_top", "r_finger_slide_bottom", "r_finger_slide_top"):
            action[:, names.index(name)] = -.2
        for _ in range(100):
            obs, reward, _, _, _ = env.step(action)
        sensor = env.scene["feet_tactile"]
        data = sensor.data
        assert tuple(data.tactile_points_w_per_sensor.shape) == (4, 4, 1024, 4)
        assert tuple(data.heatmaps.shape) == (4, 4, 32, 32)
        assert torch.isfinite(data.tactile_points_w_per_sensor).all()
        assert data.response.min() >= 0 and data.response.max() <= 1
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
        positions = env.sim.data.xpos[:, sensor.engine.source_ids].cpu().numpy()
        rotations = env.sim.data.xmat[:, sensor.engine.source_ids].cpu().numpy()
        local = foot_local_taxels()
        expected = np.einsum("esij,spj->espi", rotations, local) + positions[:, :, None, :]
        error = float(np.max(np.abs(data.tactile_points_w_per_sensor[..., :3].cpu().numpy() - expected)))
        assert error < 2e-6, error
        assert (data.response[0].max(dim=-1).values > 1.e-6).all(), data.summary()
        # Confirm all four arrays cover real sole triangles, not free space.
        native = mujoco.MjData(model)
        native.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        native.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
        native.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
        mujoco.mj_forward(model, native)
        queries, mesh_ids = [], []
        for index, patch in enumerate(FOOT_PATCHES):
            bid = model.body("robot/" + patch.link).id
            gid = next(g for g in sensor.engine.scene_target_ids if model.geom_bodyid[g] == bid and model.geom_type[g] == 7)
            geom_pos = env.sim.data.geom_xpos[0, gid].cpu().numpy()
            geom_rot = env.sim.data.geom_xmat[0, gid].cpu().numpy().reshape(3, 3)
            q = (expected[0, index] - geom_pos) @ geom_rot
            queries.append(q)
            target_index = sensor.engine.scene_target_ids.index(gid)
            mesh_ids.append(sensor.engine.scene_meshes_wp.numpy()[target_index])
        points_wp = wp.array(np.asarray(queries, dtype=np.float32), dtype=wp.vec3, device=env.device)
        meshes_wp = wp.array(mesh_ids, dtype=wp.uint64, device=env.device)
        distance_wp = wp.zeros((4, 1024), dtype=float, device=env.device)
        wp.launch(surface_distances, dim=(4, 1024), inputs=[points_wp, meshes_wp, distance_wp], device=env.device)
        surface_error = float(distance_wp.numpy().max())
        # The CAD sole is approximately planar; allow its tiny surface bevels
        # and single-precision transforms, but reject overhanging arrays.
        assert surface_error < .0001, surface_error
        flat = data.response.detach().cpu().numpy().copy()
        flat_stats = data.summary()
        save_foot_snapshot(data, output, "flat_standing")
        # Different physical ankle rotations in four independent worlds.
        action[1, names.index("l_ankle_roll")] = 1.
        action[2, names.index("l_ankle_roll")] = -1.
        action[0, names.index("r_ankle_roll")] = -1.
        action[3, names.index("r_ankle_roll")] = 1.
        for _ in range(100):
            env.step(action)
        tilted = sensor.data.response.detach().cpu().numpy().copy()
        assert np.linalg.norm(tilted[1, :2] - flat[1, :2]) > .001
        assert np.linalg.norm(tilted[2, :2] - flat[2, :2]) > .001
        right_changes = [float(np.linalg.norm(tilted[i, 2:] - flat[i, 2:])) for i in (0, 3)]
        print("Right foot map changes for opposite tilts:", right_changes, flush=True)
        for env_index in range(4):
            print("Foot diagnostics", env_index,
                  {joint: float(env.sim.data.qpos[env_index, model.jnt_qposadr[model.joint('robot/' + joint).id]].item())
                   for joint in ('l_ankle_roll', 'r_ankle_roll')},
                  sensor.data.summary(env_index), flush=True)
        assert max(right_changes) > .001, right_changes
        assert (env.scene["flexitac"].data.response.max(dim=-1).values > .01).all()
        tilt_stats = {str(i): sensor.data.summary(i) for i in range(4)}
        np.savez_compressed(output / "tilt_readings.npz", flat=flat, tilted=tilted)
        env.reset()
        reset_data = sensor.data
        assert np.linalg.norm(reset_data.response.detach().cpu().numpy() - tilted) > .001
        assert (reset_data.response.max(dim=-1).values > 1.e-6).all()
        report = {"passed": True, "shape": [4, 4, 1024, 4], "heatmaps": [4, 4, 32, 32],
                  "pitch_m": FOOT_PITCH_M, "physical_side_m": .08, "gap_m": .016,
                  "world_position_max_error_m": error, "sole_surface_max_error_m": surface_error,
                  "flat": flat_stats, "tilted": tilt_stats, "right_tilt_response_changes": right_changes,
                  "actor_shape": list(obs["actor"].shape)}
        (output / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
