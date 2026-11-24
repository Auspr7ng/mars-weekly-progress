"""Exploratory held-pose stop/reverse/turn sequence with no automatic resets."""

import bootstrap

bootstrap.configure_warp()

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from sunday_locomotion.mdp import sole_geometry
from week7_locomotion.env_cfg import make_env, agent_cfg
from week7_locomotion.mdp import arm_pose_bank
from evaluate import POSE_NAMES

SEQUENCE = (
    ("initial_stop", 2.0, [0.0, 0.0, 0.0]),
    ("forward", 4.0, [0.06, 0.0, 0.0]),
    ("stop_after_forward", 3.0, [0.0, 0.0, 0.0]),
    ("reverse", 4.0, [-0.04, 0.0, 0.0]),
    ("stop_after_reverse", 3.0, [0.0, 0.0, 0.0]),
    ("turn_left", 4.0, [0.0, 0.0, 0.12]),
    ("stop_after_left", 3.0, [0.0, 0.0, 0.0]),
    ("turn_right", 4.0, [0.0, 0.0, -0.12]),
    ("final_stop", 3.0, [0.0, 0.0, 0.0]),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5742)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 32:
        parser.error("trials must be 1..32")
    spec = json.loads((args.checkpoint.parent / "run.json").read_text())
    torch.set_num_threads(4)
    count = len(POSE_NAMES) * args.trials
    cfg = make_env(
        count,
        args.seed,
        play=False,
        progress_override=1.0,
        training_round=spec.get("training_round", 1),
    )
    cfg.week7_randomize_phase = False
    cfg.observations["actor"].enable_corruption = False
    cfg.auto_reset = False
    cfg.episode_length_s = 31.0
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
        env.command_manager.get_term("arms").override = (
            arm_pose_bank().repeat_interleave(args.trials, dim=0).tolist()
        )
        env.command_manager.get_term("twist").override = [0.0, 0.0, 0.0]
        alive = torch.ones(count, dtype=torch.bool, device=env.device)
        overall_pass = alive.clone()
        failures = {key: 0 for key in ("tilt", "height", "ground", "self", "nonfinite")}
        segments = []
        with torch.inference_mode():
            obs, _ = env.reset(seed=args.seed)
            for label, duration, requested in SEQUENCE:
                env.command_manager.get_term("twist").override = requested
                sums = torch.zeros(count, 6, device=env.device)
                samples = torch.zeros(count, device=env.device)
                settled_origin = env.scene["robot"].data.root_link_pos_w[:, :2].clone()
                for step in range(round(duration / env.step_dt)):
                    if step == round(1.2 / env.step_dt):
                        settled_origin = (
                            env.scene["robot"].data.root_link_pos_w[:, :2].clone()
                        )
                    action = policy(TensorDict(obs, batch_size=[count])).clamp(-1, 1)
                    action[~alive] = 0.0
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated | truncated
                    for reason in failures:
                        failures[reason] += int(
                            (env.termination_manager.get_term(reason) & alive).sum()
                        )
                    alive &= ~done
                    valid = alive & (step * env.step_dt >= 1.2)
                    data = env.scene["robot"].data
                    _, height = sole_geometry(env)
                    values = torch.cat(
                        (
                            data.root_link_lin_vel_b[:, :2],
                            data.root_link_ang_vel_b[:, 2:3],
                            data.root_link_lin_vel_b[:, :2].norm(dim=1, keepdim=True),
                            data.root_link_ang_vel_b[:, 2:3].abs(),
                            height.clamp(min=0).amax(dim=1, keepdim=True),
                        ),
                        dim=1,
                    )
                    sums += values * valid[:, None]
                    samples += valid
                mean = sums / samples[:, None].clamp(min=1)
                net_displacement = (
                    env.scene["robot"].data.root_link_pos_w[:, :2] - settled_origin
                ).norm(dim=1)
                passed = alive & (samples > 0)
                if requested[0] > 0:
                    passed &= mean[:, 0] >= 0.8 * requested[0]
                    passed &= mean[:, 4] <= 0.08
                elif requested[0] < 0:
                    passed &= mean[:, 0] <= 0.8 * requested[0]
                    passed &= mean[:, 4] <= 0.08
                elif requested[2]:
                    passed &= mean[:, 2] * (1 if requested[2] > 0 else -1) >= 0.8 * abs(
                        requested[2]
                    )
                    passed &= net_displacement <= 0.08
                else:
                    passed &= (
                        (mean[:, 3] <= 0.025)
                        & (mean[:, 4] <= 0.08)
                        & (mean[:, 5] <= 0.010)
                    )
                overall_pass &= passed
                per_pose = []
                for i, pose in enumerate(POSE_NAMES):
                    selected = slice(i * args.trials, (i + 1) * args.trials)
                    values = mean[selected].mean(dim=0).tolist()
                    per_pose.append(
                        dict(
                            pose=pose,
                            pass_count=int(passed[selected].sum()),
                            survival_count=int(alive[selected].sum()),
                            trials=args.trials,
                            mean_forward_m_s=values[0],
                            mean_lateral_m_s=values[1],
                            mean_yaw_rad_s=values[2],
                            mean_planar_speed_m_s=values[3],
                            mean_absolute_yaw_rad_s=values[4],
                            mean_maximum_foot_clearance_m=values[5],
                            samples_per_trial=samples[selected].tolist(),
                            mean_net_displacement_after_settle_m=float(
                                net_displacement[selected].mean()
                            ),
                            max_net_displacement_after_settle_m=float(
                                net_displacement[selected].max()
                            ),
                        )
                    )
                segments.append(
                    dict(
                        name=label,
                        seconds=duration,
                        requested_command=requested,
                        cases=per_pose,
                    )
                )
        report = dict(
            metric_version="sunday.week7.transitions.v2",
            checkpoint=str(args.checkpoint.resolve()),
            training_round=spec.get("training_round", 1),
            seed=args.seed,
            seconds=30.0,
            trials_per_pose=args.trials,
            total_trials=count,
            survived=int(alive.sum()),
            all_segment_passes=int(overall_pass.sum()),
            failure_counts=failures,
            segments=segments,
            accepted=False,
            scope="Exploratory GPU sequence with fixed held arm references; no native/hardware acceptance. Segment means are measured after 1.2s; missing samples do not pass. No within-episode arm transition is tested.",
            definition="Forward/reverse: >=80% signed speed and |yaw|<=0.08. Turn: >=80% signed yaw and net world-plane displacement<=0.08m after settling, allowing planned sway. Stop: planar speed<=0.025, |yaw|<=0.08, max foot clearance<=0.010. All require first-episode survival and valid samples.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("WEEK7_TRANSITION_CHECK " + str(args.output.resolve()), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
