"""First-episode held-pose/speed evaluation; survival alone is not walking."""

import bootstrap

bootstrap.configure_warp()

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from sunday_locomotion.mdp import contact, sole_geometry, foot_forces
from week7_locomotion.env_cfg import make_env, agent_cfg
from week7_locomotion.mdp import arm_pose_bank, foot_heading_error, inward_angles
from week7_locomotion.curriculum import TASK_VERSION

POSE_NAMES = (
    "home",
    "lift",
    "spread",
    "lift_spread",
    "full_lift",
    "full_spread",
    "asymmetric",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.06, 0.12, 0.20])
    parser.add_argument("--seed", type=int, default=1742)
    parser.add_argument(
        "--random-phase",
        action="store_true",
        help="Use uniformly randomized initial gait phase for any round",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 1 <= args.trials <= 100
        or not 3 <= args.seconds <= 60
        or any(not 0 < v <= 0.25 for v in args.speeds)
    ):
        parser.error("trials 1..100, seconds 3..60, positive forward speeds <=0.25")
    spec = json.loads((args.checkpoint.parent / "run.json").read_text())
    if spec["task_version"] != TASK_VERSION:
        raise ValueError("Evaluator requires the Week7 reference-conditioned contract")
    torch.set_num_threads(4)
    count = len(POSE_NAMES) * len(args.speeds) * args.trials
    cfg = make_env(
        count,
        args.seed,
        play=False,
        progress_override=1.0,
        training_round=spec.get("training_round", 1),
    )
    cfg.week7_randomize_phase = args.random_phase
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = args.seconds + 1.0
    cfg.rewards = {}
    cfg.terminations.pop("boundary", None)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = MjlabOnPolicyRunner(
            wrapper, asdict(agent_cfg()), log_dir=None, device="cuda:0"
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device="cuda:0")
        commands, offsets, groups = [], [], []
        bank = arm_pose_bank()
        for pose, name in enumerate(POSE_NAMES):
            for speed in args.speeds:
                start = len(commands)
                for _ in range(args.trials):
                    commands.append([speed, 0.0, 0.0])
                    offsets.append(bank[pose].tolist())
                groups.append((name, speed, start, len(commands)))
        env.command_manager.get_term("twist").override = commands
        env.command_manager.get_term("arms").override = offsets
        obs, _ = env.reset(seed=args.seed)
        robot = env.scene["robot"].data
        sole_ids = env.scene["robot"].find_sites(
            ("l_sole", "r_sole"), preserve_order=True
        )[0]
        origin = robot.root_link_pos_w.clone()
        end = origin.clone()
        alive = torch.ones(count, dtype=torch.bool, device=env.device)
        samples = torch.zeros(count, device=env.device)
        heading_sum = torch.zeros(count, 2, device=env.device)
        inward_sum = torch.zeros_like(heading_sum)
        heading_samples = torch.zeros_like(samples)
        speed_sum = torch.zeros_like(samples)
        error_sum = torch.zeros_like(samples)
        support_time = torch.zeros(count, 2, device=env.device)
        force_impulse = torch.zeros_like(support_time)
        balance_time = torch.zeros_like(samples)
        roll_sum = torch.zeros_like(samples)
        roll_square_sum = torch.zeros_like(samples)
        roll_max = torch.zeros_like(samples)
        slip_sum = torch.zeros_like(samples)
        slip_count = torch.zeros_like(samples)
        stance_ticks = torch.zeros(count, 2, dtype=torch.long, device=env.device)
        air_ticks = torch.zeros_like(stance_ticks)
        in_swing = torch.zeros(count, 2, dtype=torch.bool, device=env.device)
        peak = torch.zeros(count, 2, device=env.device)
        peak_sum = torch.zeros_like(peak)
        peak_min = torch.full_like(peak, float("inf"))
        landed_count = torch.zeros_like(stance_ticks)
        max_lateral = torch.zeros_like(samples)
        failures = {key: 0 for key in ("tilt", "height", "ground", "self", "nonfinite")}
        with torch.inference_mode():
            for step in range(round(args.seconds / env.step_dt)):
                before = robot.root_link_pos_w.clone()
                velocity = robot.root_link_lin_vel_b[:, 0].clone()
                heading_error = foot_heading_error(env)
                desired = env.command_manager.get_command("twist")[:, 0].clone()
                valid_speed = alive & (step * env.step_dt >= 1.4)
                speed_sum += velocity * valid_speed
                error_sum += (velocity - desired).square() * valid_speed
                samples += valid_speed
                heading_sum += heading_error.abs() * valid_speed[:, None]
                inward_sum += inward_angles(heading_error) * valid_speed[:, None]
                heading_samples += valid_speed
                action = policy(TensorDict(obs, batch_size=[count])).clamp(-1, 1)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                for reason in failures:
                    failures[reason] += int(
                        (env.termination_manager.get_term(reason) & alive).sum()
                    )
                end[alive] = torch.where(done[:, None], before, robot.root_link_pos_w)[
                    alive
                ]
                valid = alive & ~done
                _, lowest = sole_geometry(env)
                supported = contact(env)
                balance_valid = valid & (step * env.step_dt >= 1.4)
                balance_dt = balance_valid * env.step_dt
                support_time += supported * balance_dt[:, None]
                force_impulse += foot_forces(env).norm(dim=-1) * balance_dt[:, None]
                balance_time += balance_dt
                rotation = matrix_from_quat(robot.root_link_quat_w)
                roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
                roll_sum += roll * balance_dt
                roll_square_sum += roll.square() * balance_dt
                roll_max = torch.maximum(roll_max, roll.abs() * balance_valid)
                airborne = lowest > 0.004
                air_ticks = torch.where(airborne, air_ticks + 1, 0)
                in_swing |= air_ticks >= 3
                peak = torch.maximum(peak, lowest)
                landed = in_swing & supported & (lowest < 0.004) & valid[:, None]
                peak_sum += torch.where(landed, peak, 0.0)
                peak_min = torch.minimum(
                    peak_min, torch.where(landed, peak, float("inf"))
                )
                landed_count += landed
                peak = torch.where(landed, 0.0, peak)
                in_swing &= ~landed
                stance_ticks = torch.where(supported, stance_ticks + 1, 0)
                settled = (stance_ticks >= 5) & valid[:, None]
                slip = robot.site_lin_vel_w[:, sole_ids, :2].norm(dim=-1)
                slip_sum += (slip * settled).sum(dim=1)
                slip_count += settled.sum(dim=1)
                max_lateral = torch.maximum(
                    max_lateral,
                    torch.where(
                        valid, (robot.root_link_pos_w[:, 1] - origin[:, 1]).abs(), 0.0
                    ),
                )
                alive &= ~done
                if not alive.any():
                    break
        actual = speed_sum / samples.clamp(min=1)
        rmse = (error_sum / samples.clamp(min=1)).sqrt()
        mean_peaks = peak_sum / landed_count.clamp(min=1)
        support_fraction = support_time / balance_time[:, None].clamp(min=env.step_dt)
        load_fraction = force_impulse / force_impulse.sum(dim=1, keepdim=True).clamp(
            min=1e-6
        )
        load_imbalance = (load_fraction[:, 0] - load_fraction[:, 1]).abs()
        duty_difference = (support_fraction[:, 0] - support_fraction[:, 1]).abs()
        roll_rms = (roll_square_sum / balance_time.clamp(min=env.step_dt)).sqrt() * (
            180 / torch.pi
        )
        mean_inward = (
            inward_sum / heading_samples[:, None].clamp(min=1) * (180 / torch.pi)
        )
        peak_difference = (mean_peaks[:, 0] - mean_peaks[:, 1]).abs()
        results = []
        for pose, speed, start, stop in groups:
            selected = slice(start, stop)
            walking = alive[selected] & (actual[selected] >= 0.8 * speed)
            walking &= (landed_count[selected] >= 2).all(dim=1)
            walking &= (mean_peaks[selected] >= 0.020).all(dim=1)
            walking &= max_lateral[selected] < 0.15
            balanced = walking & (mean_peaks[selected] >= 0.030).all(dim=1)
            balanced &= peak_difference[selected] <= 0.005
            balanced &= (mean_inward[selected] <= 5).all(dim=1)
            balanced &= (
                (roll_rms[selected] <= 3)
                & (duty_difference[selected] <= 0.10)
                & (load_imbalance[selected] <= 0.10)
            )
            minimum = torch.where(
                torch.isfinite(peak_min[selected]), peak_min[selected], 0.0
            )
            results.append(
                {
                    "pose": pose,
                    "requested_forward_m_s": speed,
                    "trials": args.trials,
                    "survival_fraction": float(alive[selected].float().mean()),
                    "walking_fraction": float(walking.float().mean()),
                    "symmetric_straight_diagnostic_fraction": (
                        float(balanced.float().mean()) if pose != "asymmetric" else None
                    ),
                    "mean_per_trial_swing_peak_difference_m": float(
                        peak_difference[selected].mean()
                    ),
                    "mean_support_time_left_right_s_after_settle": support_time[
                        selected
                    ]
                    .mean(dim=0)
                    .tolist(),
                    "mean_support_duty_fraction_left_right": support_fraction[selected]
                    .mean(dim=0)
                    .tolist(),
                    "mean_per_trial_support_duty_difference": float(
                        duty_difference[selected].mean()
                    ),
                    "mean_contact_force_magnitude_impulse_left_right_Ns": force_impulse[
                        selected
                    ]
                    .mean(dim=0)
                    .tolist(),
                    "mean_contact_force_magnitude_fraction_left_right": load_fraction[
                        selected
                    ]
                    .mean(dim=0)
                    .tolist(),
                    "mean_per_trial_contact_load_magnitude_imbalance": float(
                        load_imbalance[selected].mean()
                    ),
                    "mean_signed_body_roll_deg": float(
                        (
                            roll_sum[selected]
                            / balance_time[selected].clamp(min=env.step_dt)
                        ).mean()
                        * (180 / torch.pi)
                    ),
                    "mean_body_roll_rms_deg": float(roll_rms[selected].mean()),
                    "max_absolute_body_roll_deg": float(
                        roll_max[selected].max() * (180 / torch.pi)
                    ),
                    "mean_actual_forward_m_s_after_settle": float(
                        actual[selected].mean()
                    ),
                    "mean_absolute_foot_heading_left_right_deg": (
                        heading_sum[selected]
                        / heading_samples[selected, None].clamp(min=1)
                        * (180 / torch.pi)
                    )
                    .mean(dim=0)
                    .tolist(),
                    "mean_inward_toe_angle_left_right_deg": (
                        inward_sum[selected]
                        / heading_samples[selected, None].clamp(min=1)
                        * (180 / torch.pi)
                    )
                    .mean(dim=0)
                    .tolist(),
                    "post_settle_sample_count_per_trial": samples[selected].tolist(),
                    "mean_forward_rmse_m_s": float(rmse[selected].mean()),
                    "mean_completed_swing_peak_left_right_m": mean_peaks[selected]
                    .mean(dim=0)
                    .tolist(),
                    "minimum_completed_swing_peak_left_right_m": minimum.amin(
                        dim=0
                    ).tolist(),
                    "mean_completed_steps_left_right": landed_count[selected]
                    .float()
                    .mean(dim=0)
                    .tolist(),
                    "mean_settled_support_slip_m_s": float(
                        (slip_sum[selected] / slip_count[selected].clamp(min=1)).mean()
                    ),
                    "mean_forward_displacement_m": float(
                        (end[selected, 0] - origin[selected, 0]).mean()
                    ),
                    "max_lateral_drift_m": float(max_lateral[selected].max()),
                }
            )
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "task_version": TASK_VERSION,
            "training_round": spec.get("training_round", 1),
            "randomized_start_phase": args.random_phase,
            "seed": args.seed,
            "seconds": args.seconds,
            "cases": results,
            "failure_counts": failures,
            "accepted": False,
            "scope": "Exploratory first-episode forward matrix. No native-physics, turning, perturbation or hardware acceptance.",
            "walking_definition": "Survive; mean actual speed >=80% of command after 1.4s; >=2 completed swings per foot; mean swing peaks >=20mm both feet; lateral drift <0.15m. Failed trials contribute no walking pass.",
            "symmetric_straight_diagnostic_definition": "For symmetric held-arm poses only: existing walking gate; both mean completed peaks >=30mm; per-trial L/R peak gap <=5mm; inward toe yaw <=5deg each; body roll RMS <=3deg; support duty fraction difference <=0.10; contact-force-magnitude share difference <=0.10. Diagnostic targets, not deployment acceptance.",
            "load_metric_note": "Integral of contact force magnitude from the existing smoothed sensor; a load-transfer proxy, not center-of-pressure or a complete dynamic balance criterion.",
            "new_fast_target_m_s": 0.20,
            "published_R25_limit_m_s": 0.12,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("WEEK7_EVALUATION " + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
