"""GPU contract check for held references, masked arm actions and finite physics."""

import bootstrap

bootstrap.configure_warp()

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from week7_locomotion.curriculum import ARM_NAMES
from week7_locomotion.symmetry import reflect_group, augment, mirror_arm_offsets
from sunday_locomotion.symmetry import mapping, reflect_actions
from tensordict import TensorDict
from week7_locomotion.env_cfg import make_env
from week7_locomotion.mdp import (
    arm_pose_bank,
    foot_heading_error,
    toe_in_cost,
    foot_heading_cost,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=bootstrap.WEEK / "results/environment_check.json"
    )
    parser.add_argument(
        "--round", dest="training_round", type=int, choices=(1, 2, 3, 4, 5), default=1
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(
        make_env(
            7,
            seed=742,
            play=True,
            progress_override=1.0,
            training_round=args.training_round,
        ),
        device="cuda:0",
    )
    try:
        arms = env.command_manager.get_term("arms")
        arms.override = arm_pose_bank("cuda:0")
        env.command_manager.get_term("twist").override = [0.0, 0.0, 0.0]
        obs, _ = env.reset(seed=742)
        checks = {
            "actor_shape": obs["actor"].shape == (7, 510),
            "critic_shape": obs["critic"].shape == (7, 108),
            "reference_history_suffix": bool(
                torch.allclose(obs["actor"][:, -10:], arms.offsets)
            ),
            "spawn_at_requested_pose": bool(
                torch.allclose(
                    env.scene["robot"].data.joint_pos[:, arms.ids],
                    arms.targets,
                    atol=1e-6,
                )
            ),
        }
        headings = foot_heading_error(env)
        checks["home_toe_axis_near_forward"] = bool((headings[0].abs() < 0.04).all())
        checks["finite_heading_costs"] = bool(
            torch.isfinite(toe_in_cost(env)).all()
            and torch.isfinite(foot_heading_cost(env)).all()
        )
        if args.training_round >= 3:
            from week7_locomotion.balance import symmetric_straight_gate

            m = mapping(env)
            for group in ("actor", "critic"):
                probe = torch.randn_like(obs[group])
                mirrored = reflect_group(probe, group, m)
                restored = reflect_group(mirrored, group, m)
                checks[f"{group}_mirror_involution"] = bool(
                    torch.allclose(restored, probe, atol=1e-6)
                )
                prefix, history = (460, 5) if group == "actor" else (98, 1)
                refs = probe[:, prefix:].reshape(7, history, 10)
                checks[f"{group}_reference_mirror"] = bool(
                    torch.equal(
                        mirrored[:, prefix:], mirror_arm_offsets(refs).flatten(-2)
                    )
                )
            probe = torch.randn(7, 22, device=env.device)
            checks["action_mirror_involution"] = bool(
                torch.allclose(
                    reflect_actions(reflect_actions(probe, m), m), probe, atol=1e-6
                )
            )
            env.command_manager.get_term("twist").value[:] = torch.tensor(
                [0.12, 0.0, 0.0], device=env.device
            )
            env.command_manager.get_term("twist").motion_gate[:] = 1.0
            env.episode_length_buf[:] = 100
            gate = symmetric_straight_gate(env)
            checks["symmetry_gate_rejects_asymmetric_arms"] = gate.tolist() == [
                True
            ] * 6 + [False]
            env.command_manager.get_term("twist").value[:, 2] = 0.20
            checks["symmetry_gate_rejects_turns"] = not bool(
                symmetric_straight_gate(env).any()
            )
            env.reset(seed=742)
            td = TensorDict(obs, batch_size=[7])
            augmented, _ = augment(env, obs=td)
            checks["new_schema_augmentation_shape"] = augmented["actor"].shape == (
                14,
                510,
            )
            # Check all randomized phases, not only the fixed initial phase.
            env.cfg.week7_randomize_phase = True
            env.reset(seed=742)
            phases = env.command_manager.get_term("twist").gait_phase
            checks["randomized_start_phase"] = bool(
                phases.std() > 0.05 and ((phases >= 0) & (phases < 1)).all()
            )
        action = env.action_manager.get_term("residual")
        arm_columns = [i for i, n in enumerate(action.names) if n in ARM_NAMES]
        actions = torch.zeros(7, 22, device="cuda:0")
        actions[:, arm_columns] = 1.0
        action.process_actions(actions)
        checks["arm_policy_outputs_ignored"] = bool(
            torch.equal(action.targets[:, arms.ids], arms.targets)
        )
        checks["twelve_leg_outputs"] = len(action.leg_columns) == 12
        checks["reference_is_soft_limit_bounded"] = bool(
            (
                (
                    arms.targets
                    >= env.scene["robot"].data.soft_joint_pos_limits[:, arms.ids, 0]
                )
                & (
                    arms.targets
                    <= env.scene["robot"].data.soft_joint_pos_limits[:, arms.ids, 1]
                )
            ).all()
        )
        finite = True
        for _ in range(12):
            obs, reward, _, _, _ = env.step(actions)
            finite &= bool(
                torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
            )
        checks["finite_full_pose_physics"] = finite
        # Reset a subset: per-environment override lookup must preserve its row.
        selected = torch.tensor([1, 6], device="cuda:0")
        env.reset(env_ids=selected)
        checks["subset_pose_reset"] = bool(
            torch.allclose(arms.offsets[selected], arms.override[selected])
        )
        report = {
            "training_round": args.training_round,
            "initial_foot_heading_deg": (headings * (180 / torch.pi)).tolist(),
            "checks": checks,
            "passed": all(checks.values()),
            "scope": "Schema and target ownership, not balance or gait acceptance",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("WEEK7_ENVIRONMENT_CHECK " + json.dumps(report), flush=True)
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
