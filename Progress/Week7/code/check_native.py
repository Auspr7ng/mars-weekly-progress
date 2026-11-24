"""Independent native-MuJoCo/CPU-actor held-pose check for Week7 checkpoints."""

import bootstrap

bootstrap.configure_warp()

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from sunday_locomotion.native_observations import NativeHistory
from sunday_locomotion.cadence import native_period
from week7_locomotion.env_cfg import make_env, agent_cfg
from week7_locomotion.mdp import arm_pose_bank
from evaluate import POSE_NAMES


class Week7NativeHistory(NativeHistory):
    def next_phase(self, command):
        faster = np.clip((command[0] - 0.15) / 0.05, 0.0, 1.0)
        period = native_period(command) - 0.15 * faster
        return np.float32((self.gait_phase + np.float32(0.02) / period) % 1.0)

    def push_reference(self, data, command, last, step, references, heading=0.0):
        prefix = self.push(data, command, last, step, heading)
        return np.concatenate([prefix, np.tile(references, 5)]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--poses",
        nargs="+",
        choices=POSE_NAMES,
        default=["home", "lift_spread", "asymmetric"],
    )
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.06, 0.12, 0.20])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 30 or any(not 0 < v <= 0.20 for v in args.speeds):
        parser.error(
            "seconds must be 3..30, forward speeds must be positive and <=0.20"
        )
    torch.set_num_threads(4)
    spec = json.loads((args.checkpoint.parent / "run.json").read_text())
    cfg = make_env(
        1,
        seed=4742,
        play=True,
        progress_override=1.0,
        training_round=spec.get("training_round", 1),
    )
    cfg.week7_randomize_phase = False
    cfg.observations["actor"].enable_corruption = False
    cfg.rewards = {}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = MjlabOnPolicyRunner(
            wrapper, asdict(agent_cfg()), log_dir=None, device="cuda:0"
        )
        runner.load(str(args.checkpoint))
        actor = runner.get_inference_policy(device="cpu")
        robot = env.scene["robot"]
        action = env.action_manager.get_term("residual")
        names = list(robot.joint_names)
        default = robot.data.default_joint_pos[0].cpu().numpy().copy()
        limits = robot.data.soft_joint_pos_limits[0].cpu().numpy()
        ids = action.ids.cpu().numpy()
        scales, speeds = action.scales.cpu().numpy(), action.speeds.cpu().numpy()
        arm_ids = env.command_manager.get_term("arms").ids.cpu().numpy()
        model = env.sim.mj_model
        actuator_joints = [
            names.index(model.actuator(i).name.removeprefix("robot/"))
            for i in range(model.nu)
        ]
        root = model.body("robot/base_link").id
        feet = [model.body("robot/" + side + "_foot_link_1").id for side in ("l", "r")]
        sole_ids = [model.site("robot/" + side + "_sole").id for side in ("l", "r")]
        substeps = round(0.02 / model.opt.timestep)
        cases, parity = [], []
        for pose in args.poses:
            references = arm_pose_bank()[POSE_NAMES.index(pose)].numpy()
            for requested in args.speeds:
                env.command_manager.get_term("arms").override = references.tolist()
                env.command_manager.get_term("twist").override = [0.0, 0.0, 0.0]
                with torch.inference_mode():
                    obs, _ = env.reset(seed=4742)
                data = mujoco.MjData(model)
                data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                targets = default.copy()
                targets[arm_ids] += references
                data.ctrl[:] = targets[actuator_joints]
                mujoco.mj_forward(model, data)
                history = Week7NativeHistory(
                    model, default, names, len(ids), revision=26
                )
                last, command = np.zeros(len(ids)), np.zeros(3)
                initial = history.push_reference(data, command, last, 0, references)
                difference = float(
                    np.max(np.abs(initial - obs["actor"][0].cpu().numpy()))
                )
                parity.append(difference)
                if difference > 2e-5:
                    raise ValueError(
                        f"Native/GPU initial observation mismatch: {difference}"
                    )
                # Begin a fresh history after the parity probe.
                history = Week7NativeHistory(
                    model, default, names, len(ids), revision=26
                )
                origin = data.qpos[:3].copy()
                measured, peak_samples = [], [[], []]
                peak, air_ticks, in_swing = (
                    np.zeros(2),
                    np.zeros(2, dtype=int),
                    np.zeros(2, dtype=bool),
                )
                counters = dict(tilt=0, height=0, ground=0, self=0)
                reasons, max_drift = [], 0.0
                with torch.inference_mode():
                    for step in range(round(args.seconds / 0.02)):
                        observation = history.push_reference(
                            data, command, last, step, references
                        )
                        out = (
                            actor(
                                TensorDict(
                                    {"actor": torch.from_numpy(observation[None])},
                                    batch_size=[1],
                                )
                            )[0]
                            .clamp(-1, 1)
                            .numpy()
                        )
                        if not np.isfinite(out).all():
                            reasons = ["nonfinite_action"]
                            break
                        desired = default[ids] + out * scales
                        desired = targets[ids] + 0.3 * (desired - targets[ids])
                        desired = np.clip(
                            desired,
                            targets[ids] - speeds * 0.02,
                            targets[ids] + speeds * 0.02,
                        )
                        targets[ids] = np.clip(desired, limits[ids, 0], limits[ids, 1])
                        targets[arm_ids] = np.clip(
                            default[arm_ids] + references,
                            limits[arm_ids, 0],
                            limits[arm_ids, 1],
                        )
                        data.ctrl[:] = targets[actuator_joints]
                        bad_ground = bad_self = False
                        contact_force = np.zeros(2)
                        for _ in range(substeps):
                            mujoco.mj_step(model, data)
                            for ci in range(data.ncon):
                                contact = data.contact[ci]
                                b1, b2 = int(model.geom_bodyid[contact.geom1]), int(
                                    model.geom_bodyid[contact.geom2]
                                )
                                plane = (
                                    model.geom_type[contact.geom1]
                                    == mujoco.mjtGeom.mjGEOM_PLANE
                                    or model.geom_type[contact.geom2]
                                    == mujoco.mjtGeom.mjGEOM_PLANE
                                )
                                force = np.zeros(6)
                                mujoco.mj_contactForce(model, data, ci, force)
                                if plane:
                                    for foot, body in enumerate(feet):
                                        if body in (b1, b2):
                                            contact_force[foot] += (
                                                abs(force[0]) / substeps
                                            )
                                    if b1 not in feet and b2 not in feet:
                                        bad_ground |= (
                                            abs(force[0])
                                            > 0.02 * cfg.sunday_mass * 9.81
                                        )
                                elif contact.dist < 0:
                                    bad_self |= model.geom(
                                        contact.geom1
                                    ).name.startswith("robot/") and model.geom(
                                        contact.geom2
                                    ).name.startswith(
                                        "robot/"
                                    )
                        mujoco.mj_forward(model, data)
                        last = out.copy()
                        command += np.clip(
                            np.array([requested, 0.0, 0.0]) - command,
                            -np.array([0.5, 0.5, 0.8]) * 0.02,
                            np.array([0.5, 0.5, 0.8]) * 0.02,
                        )
                        velocity = np.zeros(6)
                        mujoco.mj_objectVelocity(
                            model, data, mujoco.mjtObj.mjOBJ_XBODY, root, velocity, 1
                        )
                        if step * 0.02 >= 1.4:
                            measured.append(float(velocity[3]))
                        rotation = data.site_xmat[sole_ids].reshape(2, 3, 3)
                        lowest = (
                            data.site_xpos[sole_ids, 2]
                            - np.abs(rotation[:, 2, 0]) * 0.0525
                            - np.abs(rotation[:, 2, 1]) * 0.09
                        )
                        air_ticks = np.where(lowest > 0.004, air_ticks + 1, 0)
                        in_swing |= air_ticks >= 3
                        peak = np.maximum(peak, lowest)
                        landed = (
                            in_swing
                            & (contact_force > 0.03 * cfg.sunday_mass * 9.81)
                            & (lowest < 0.004)
                        )
                        for foot in range(2):
                            if landed[foot]:
                                peak_samples[foot].append(float(peak[foot]))
                        peak = np.where(landed, 0.0, peak)
                        in_swing &= ~landed
                        max_drift = max(max_drift, abs(data.qpos[1] - origin[1]))
                        cosine = data.xmat[root].reshape(3, 3)[2, 2]
                        bad = dict(
                            tilt=cosine < 0.5,
                            height=data.qpos[2] < 0.55 * cfg.sunday_h0,
                            ground=bad_ground,
                            self=bad_self,
                        )
                        for key, flag in bad.items():
                            counters[key] = counters[key] + 1 if flag else 0
                        reasons = [
                            key
                            for key, count in counters.items()
                            if count >= (5 if key in ("tilt", "height") else 2)
                        ]
                        if cosine < 0.173648:
                            reasons.append("extreme_tilt")
                        if not np.isfinite(data.qpos).all():
                            reasons.append("nonfinite_state")
                        if reasons:
                            break
                peaks = [float(np.mean(x)) if x else 0.0 for x in peak_samples]
                speed = float(np.mean(measured)) if measured else None
                passed = (
                    not reasons
                    and speed is not None
                    and speed >= 0.8 * requested
                    and all(len(x) >= 2 for x in peak_samples)
                    and min(peaks) >= 0.020
                    and max_drift < 0.15
                )
                cases.append(
                    dict(
                        pose=pose,
                        requested_forward_m_s=requested,
                        seconds_completed=(step + 1) * 0.02,
                        termination_reasons=reasons,
                        mean_actual_forward_m_s_after_settle=speed,
                        completed_steps_left_right=[len(x) for x in peak_samples],
                        mean_completed_swing_peak_left_right_m=peaks,
                        max_lateral_drift_m=float(max_drift),
                        exploratory_walking_pass=bool(passed),
                    )
                )
        report = dict(
            checkpoint=str(args.checkpoint.resolve()),
            physics="native MuJoCo",
            actor_device="cpu",
            seconds_requested=args.seconds,
            fixed_initial_phase=True,
            initial_observation_max_abs_error=max(parity),
            cases=cases,
            accepted=False,
            scope="Independent single-start held-pose forward check. Not a hardware release or full native balance acceptance.",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("WEEK7_NATIVE_CHECK " + str(args.output.resolve()), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
