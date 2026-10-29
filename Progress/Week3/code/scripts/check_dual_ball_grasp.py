"""GPU regression for simultaneous grasps, empty close, offsets and isolated resets."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

from sunday_a1_mjlab.ball_grasp import BALL_ENTITIES, BALL_RADIUS, CONTACT_SENSORS, GRIP_TARGET, BallGraspBackend
from sunday_a1_mjlab.flexitac import FINGERS, local_taxels
from sunday_a1_mjlab.tactile_display import physical_heatmap, save_snapshot


def check_tactile_contact_planes(backend):
    """Compare samples with the actual ball-facing box faces, not a copied sign."""
    points = local_taxels()
    report = {}
    for index, finger in enumerate(FINGERS):
        hand = "left" if index < 2 else "right"
        body = backend.model.body(backend.prefix + finger.link).id
        pad = backend.model.geom(backend.prefix + "grip_pad_" + finger.name)
        assert pad.type[0] == mujoco.mjtGeom.mjGEOM_BOX
        rotation = backend.data.xmat[body].reshape(3, 3)
        ball = backend.data.geom(BALL_ENTITIES[hand] + "/sphere").xpos
        direction = rotation.T @ (ball - backend.data.geom_xpos[pad.id])
        facing = np.sign(direction[1])
        assert facing != 0
        expected_y = pad.pos[1] + facing * pad.size[1]
        error = float(np.abs(points[index, :, 1] - expected_y).max())
        assert error < 1e-7, (finger.name, "Tactile plane misses contact face", error)
        assert finger.inward == facing, (finger.name, "Normal points away from ball")
        display_z = physical_heatmap(points[index, :, 2], index)
        assert (np.diff(display_z[:, 0]) < 0).all(), (finger.name, "Root/tip reversed")
        report[finger.name] = {"contact_plane_error_m": error, "normal_toward_ball": True,
                               "display_root_at_top": True}
    return report


def main():
    output = PROJECT.parent / "results/ball_grasp"
    output.mkdir(parents=True, exist_ok=True)
    grasp = BallGraspBackend()
    backend = grasp.runtime
    targets = [GRIP_TARGET] * 4
    report = {"tactile_contact_planes": check_tactile_contact_planes(backend)}

    def step(n):
        for _ in range(n):
            result = backend.step(targets)
        assert np.isfinite(backend.data.qpos).all()
        assert result.response.isfinite().all()
        return result

    def held(hand):
        sensor = backend.env.scene[CONTACT_SENSORS[hand]].data
        assert (sensor.found > 0).all(), (hand, sensor.found, backend.actual_positions())
        force = sensor.force[0].detach().cpu().numpy()
        assert abs(force[:, 2].sum() - .04 * 9.81) < .025, force
        return force.tolist()

    def check_reset_isolation(hand, offset):
        other = "right" if hand == "left" else "left"
        entity = backend.env.scene[BALL_ENTITIES[other]]
        before_ball = entity.data.root_link_pose_w.clone()
        before_vel = entity.data.root_link_vel_w.clone()
        ids = grasp.hand_joint_ids[other]
        before_joints = backend.robot.data.joint_pos[:, ids].clone()
        before_clock = backend.data.time
        position = grasp.reset_hand(hand, offset)
        assert (entity.data.root_link_pose_w == before_ball).all()
        assert (entity.data.root_link_vel_w - before_vel).abs().max() < 1e-6
        assert (backend.robot.data.joint_pos[:, ids] == before_joints).all()
        assert backend.data.time == before_clock
        return position.tolist()

    try:
        assert backend.model.neq == 0
        for entity in BALL_ENTITIES.values():
            assert backend.model.jnt_type[backend.model.joint(entity + "/free").id] == 0
        result = step(100)
        start_positions = {hand: backend.data.body(entity + "/ball").xpos.copy()
                           for hand, entity in BALL_ENTITIES.items()}
        reference_response = result.response[0].detach().cpu().numpy().copy()
        result = step(200)
        report["both_held"] = {}
        for hand, entity in BALL_ENTITIES.items():
            drift = float(np.linalg.norm(backend.data.body(entity + "/ball").xpos - start_positions[hand]))
            assert drift < .005, (hand, drift)
            report["both_held"][hand] = {"forces_world_N": held(hand), "drift_4s_m": drift}
        assert (result.response.max(dim=-1).values > .01).all()
        initial_summary = result.summary()
        report["initial_tactile"] = initial_summary
        for left, right in (("left_bottom", "right_bottom"), ("left_top", "right_top")):
            a, b = initial_summary[left], initial_summary[right]
            assert abs(a["mean"] - b["mean"]) < .005, (left, right, a, b)
            assert abs(a["min_distance_mm"] - b["min_distance_mm"]) < .1, (left, right, a, b)
        save_snapshot(result, output, "dual_held_heatmaps")
        # Release one hand; the other continues supporting its own ball.
        targets[:2] = [.03] * 2
        result = step(100)
        assert not backend.env.scene[CONTACT_SENSORS["left"]].data.found.any()
        held("right")
        assert result.response[0, :2].max().item() == 0
        assert abs(backend.data.body("ball/ball").xpos[2] - BALL_RADIUS) < .002
        targets[:2] = [0.] * 2
        result = step(100)
        assert (result.response[0, :2].max(dim=-1).values > .01).all(), "Empty left close lost finger response"
        held("right")
        report["left_empty_close"] = result.summary()
        save_snapshot(result, output, "left_empty_right_ball_heatmaps")
        # Reposition only left ball. Heatmap moves; right state is untouched by reset.
        report["left_reset_position_m"] = check_reset_isolation("left", (4., 0., 10.))
        targets[:2] = [GRIP_TARGET] * 2
        result = step(100)
        held("left")
        held("right")
        shifted = result.response[0].detach().cpu().numpy().copy()
        assert np.linalg.norm(shifted[:2] - reference_response[:2]) > 1
        report["heatmap_change_norm"] = float(np.linalg.norm(shifted[:2] - reference_response[:2]))
        save_snapshot(result, output, "shifted_left_heatmaps")
        targets[2:] = [.03] * 2
        result = step(100)
        assert result.response[0, 2:].max().item() == 0
        held("left")
        targets[2:] = [0.] * 2
        result = step(100)
        assert (result.response[0, 2:].max(dim=-1).values > .01).all(), "Empty right close lost finger response"
        report["right_empty_close"] = result.summary()
        report["right_reset_position_m"] = check_reset_isolation("right", (-4., 2., -8.))
        targets[2:] = [GRIP_TARGET] * 2
        result = step(100)
        held("left")
        held("right")
        report["passed"] = True
        report["scene_geometry_count"] = len(backend.sensor.engine.scene_target_ids)
        (output / "dual_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        grasp.close()


if __name__ == "__main__":
    main()
