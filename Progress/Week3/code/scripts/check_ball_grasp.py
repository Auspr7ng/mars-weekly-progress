"""Regression: two real finger contacts hold a free ball; opening releases it."""
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

from sunday_a1_mjlab.ball_grasp import BALL_MASS, BALL_RADIUS, GRIP_TARGET, TASK_ID
from sunday_a1_mjlab.tactile_backend import MjlabTactileBackend
from sunday_a1_mjlab.tactile_display import save_snapshot


def main():
    output = PROJECT.parent / "results/ball_grasp"
    output.mkdir(parents=True, exist_ok=True)
    backend = MjlabTactileBackend(task=TASK_ID)
    model = backend.model
    assert model.jnt_type[model.joint("ball/free").id] == 0  # mjJNT_FREE
    assert model.neq == 0, "Grasp must not use a weld/equality constraint"
    assert model.opt.gravity[2] < -9
    states, forces, responses = [], [], []
    try:
        print("Initial ball:", backend.data.body("ball/ball").xpos, flush=True)
        # First 1 s allows joint controllers to settle; then hold for 4 s.
        for step in range(250):
            result = backend.step([GRIP_TARGET, GRIP_TARGET, .03, .03])
            contact = backend.env.scene["grip_contacts"].data
            states.append(backend.data.body("ball/ball").xpos.copy())
            forces.append(contact.force[0].detach().cpu().numpy().copy())
            responses.append(result.response[0].detach().cpu().numpy().copy())
            if step % 50 == 49:
                print("Hold", step, states[-1], contact.found.tolist(), forces[-1], flush=True)
        held = np.array(states[50:])
        assert np.isfinite(held).all()
        drift = float(np.linalg.norm(held - held[0], axis=1).max())
        assert drift < .005, f"Ball slipped {drift} m"
        assert (contact.found > 0).all(), "Both left pads must contact the ball"
        assert (np.linalg.norm(forces[-1], axis=-1) > .1).all()
        assert np.all(np.max(responses[-1][:2], axis=-1) > .01)
        assert np.max(responses[-1][2:]) == 0
        points = result.tactile_points_w_per_sensor[0].detach().cpu().numpy()
        expected_distance = np.linalg.norm(points[..., :3] - states[-1], axis=-1) - BALL_RADIUS
        actual_distance = result.distance_m[0].detach().cpu().numpy()
        near_sphere = expected_distance < .0015
        assert near_sphere.any()
        assert np.max(np.abs(expected_distance[near_sphere] - actual_distance[near_sphere])) < 1e-6
        assert abs(np.sum(forces[-1], axis=0)[2] - BALL_MASS * 9.81) < .02
        held_height = float(states[-1][2])
        save_snapshot(result, output, "held_heatmaps")
        hold_forces = forces[-1].tolist()
        # No state writes, external forces, or attachments in either phase.
        for step in range(100):
            result = backend.step([.03] * 4)
            states.append(backend.data.body("ball/ball").xpos.copy())
            forces.append(backend.env.scene["grip_contacts"].data.force[0].detach().cpu().numpy().copy())
            responses.append(result.response[0].detach().cpu().numpy().copy())
        released_height = float(states[-1][2])
        assert held_height - released_height > .03, "Opening must make the ball drop"
        assert abs(released_height - BALL_RADIUS) < .002, "Released ball should land on floor"
        assert not backend.env.scene["grip_contacts"].data.found.any()
        assert result.response.max().item() == 0
        save_snapshot(result, output, "released_heatmaps")
        # Reset restores both free object and robot, and is repeatable.
        backend.env.reset()
        backend._sync_display()
        reset_position = backend.data.body("ball/ball").xpos.copy()
        assert np.linalg.norm(reset_position - states[0]) < .005
        report = {"task": TASK_ID, "backend": "mjlab / MuJoCo-Warp CUDA",
                  "ball_radius_m": BALL_RADIUS, "ball_mass_kg": BALL_MASS,
                  "gravity": model.opt.gravity.tolist(), "ball_free_joint": True,
                  "equality_constraints": model.neq, "hold_duration_s": 4,
                  "hold_max_drift_m": drift, "held_height_m": held_height,
                  "released_height_m": released_height, "hold_contact_forces_world_N": hold_forces,
                  "contact_primary_names": backend.env.scene["grip_contacts"].primary_names,
                  "reset_ball_position_m": reset_position.tolist(), "passed": True}
        np.savez_compressed(output / "rollout.npz", time_s=np.arange(1, 351) * .02,
                            ball_position_m=np.array(states),
                            contact_forces_world_N=np.array(forces),
                            tactile_response=np.array(responses))
        (output / "check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
