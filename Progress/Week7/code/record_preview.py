"""Record actual R26 GPU rollouts with full-body and foot-closeup views."""

import bootstrap

bootstrap.configure_warp()

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from sunday_locomotion.mdp import sole_geometry
from week7_locomotion.env_cfg import make_env, agent_cfg
from week7_locomotion.mdp import arm_pose_bank, foot_heading_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 15:
        parser.error("seconds per scene must be 3..15")
    torch.set_num_threads(4)
    spec = json.loads((args.checkpoint.parent / "run.json").read_text())
    training_round = spec.get("training_round", 1)
    model_label = f"R26 R{training_round}-{args.checkpoint.stem.removeprefix('model_')}"
    cfg = make_env(
        1, seed=1742, play=False, progress_override=1.0, training_round=training_round
    )
    cfg.week7_randomize_phase = False
    cfg.observations["actor"].enable_corruption = False
    cfg.auto_reset = False
    cfg.rewards = {}
    cfg.episode_length_s = args.seconds + 1.0
    cfg.terminations.pop("boundary", None)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    renderer = writer = None
    trace = []
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = MjlabOnPolicyRunner(
            wrapper, asdict(agent_cfg()), log_dir=None, device="cuda:0"
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device="cuda:0")
        model = env.sim.mj_model
        model.vis.global_.offwidth, model.vis.global_.offheight = 640, 640
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=640, width=640)
        full, legs = mujoco.MjvCamera(), mujoco.MjvCamera()
        for camera in (full, legs):
            camera.azimuth, camera.elevation = 125, -12
        full.distance, legs.distance = 0.95, 0.48
        legs.elevation = -45
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(args.output),
            size=(1280, 720),
            fps=25,
            codec="libx264",
            macro_block_size=1,
            output_params=["-crf", "20"],
        )
        writer.send(None)
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
        scenes = (("Home arms", 0), ("Raised arms", 1), ("Raised and spread arms", 3))
        for label, pose in scenes:
            env.command_manager.get_term("arms").override = arm_pose_bank()[
                pose
            ].tolist()
            env.command_manager.get_term("twist").override = [0.20, 0.0, 0.0]
            with torch.inference_mode():
                obs, _ = env.reset(seed=1742)
            last_frame = None
            with torch.inference_mode():
                for step in range(round(args.seconds / env.step_dt)):
                    action = policy(TensorDict(obs, batch_size=[1])).clamp(-1, 1)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = bool((terminated | truncated).item())
                    reasons = (
                        [
                            name
                            for name in env.termination_manager.active_terms
                            if bool(env.termination_manager.get_term(name).any())
                        ]
                        if done
                        else []
                    )
                    data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                    data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                    mujoco.mj_forward(model, data)
                    velocity = float(env.scene["robot"].data.root_link_lin_vel_b[0, 0])
                    sole_positions, lowest = sole_geometry(env)
                    clearance = lowest[0].cpu().tolist()
                    toe_yaw = (
                        (foot_heading_error(env)[0] * (180 / torch.pi)).cpu().tolist()
                    )
                    trace.append(
                        {
                            "model_label": model_label,
                            "scene": label,
                            "time_s": (step + 1) * 0.02,
                            "requested_m_s": 0.20,
                            "actual_m_s": velocity,
                            "whole_sole_clearance_left_right_m": clearance,
                            "foot_heading_error_left_right_deg": toe_yaw,
                            "terminated": done,
                            "reasons": reasons,
                        }
                    )
                    if step % 2 == 0 or done:
                        full.lookat[:] = data.qpos[:3] + [0, 0, 0.035]
                        legs.lookat[:] = sole_positions[0].mean(dim=0).cpu().numpy() + [
                            0,
                            0,
                            0.055,
                        ]
                        renderer.update_scene(data, camera=full)
                        left = renderer.render().copy()
                        renderer.update_scene(data, camera=legs)
                        right = renderer.render().copy()
                        canvas = Image.new("RGB", (1280, 720), (20, 25, 31))
                        canvas.paste(Image.fromarray(left), (0, 45))
                        canvas.paste(Image.fromarray(right), (640, 45))
                        draw = ImageDraw.Draw(canvas)
                        draw.text(
                            (18, 10),
                            f"{model_label}: {label}   Requested 0.20 m/s   Actual {velocity:+.3f} m/s",
                            font=font,
                            fill="white",
                        )
                        footer = f"GPU MuJoCo  t={(step + 1) * .02:.2f}s  |  Toe yaw L/R: {toe_yaw[0]:+.1f}/{toe_yaw[1]:+.1f} deg  |  Full body / Feet"
                        if done:
                            footer = "STOP: " + ", ".join(reasons)
                        draw.text(
                            (18, 691),
                            footer,
                            font=small,
                            fill=(255, 180, 120) if done else "white",
                        )
                        last_frame = np.asarray(canvas)
                        writer.send(last_frame)
                        if step == 100:
                            canvas.save(
                                args.output.with_name(args.output.stem + f"_{pose}.jpg")
                            )
                    if done:
                        for _ in range(25):
                            writer.send(last_frame)
                        break
            print("PREVIEW_SCENE_DONE " + label, flush=True)
        args.output.with_suffix(".jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in trace), encoding="utf-8"
        )
        print("WEEK7_PREVIEW " + str(args.output.resolve()), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
