"""Held arm references and training-only gait shaping for Week7.

No root actuation, contact overrides, mass changes, or torque increases are used.
Arm references are fixed within each episode in this first curriculum.
"""

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from sunday_locomotion.mdp import ResidualAction, ResidualActionCfg, Twist, TwistCfg
from sunday_locomotion import mdp as base
from sunday_locomotion.control_mdp import foot_targets, gait_weight
from .curriculum import ARM_NAMES, env_targets


def arm_pose_bank(device="cpu"):
    """Offsets in ARM_NAMES order, including a deliberately asymmetric pose."""
    bank = torch.zeros(7, 10, device=device)
    bank[1, 0], bank[1, 5] = 0.35, -0.35
    bank[2, 1], bank[2, 6] = 0.35, 0.35
    bank[3, 0], bank[3, 5], bank[3, 1], bank[3, 6] = 0.35, -0.35, 0.35, 0.35
    bank[4, 0], bank[4, 5] = 0.70, -0.70
    bank[5, 1], bank[5, 6] = 0.65, 0.65
    bank[6, 0], bank[6, 5], bank[6, 1], bank[6, 6] = 0.15, -0.50, 0.35, 0.10
    return bank


@dataclass(kw_only=True)
class ArmReferenceCfg(CommandTermCfg):
    def build(self, env):
        return ArmReference(self, env)


class ArmReference(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in ARM_NAMES], device=self.device
        )
        self.offsets = torch.zeros(self.num_envs, 10, device=self.device)
        self.targets = self.robot.data.default_joint_pos[:, self.ids].clone()
        self.pose_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.override = None

    @property
    def command(self):
        return self.offsets

    def _resample_command(self, env_ids):
        count = len(env_ids)
        level = env_targets(self._env)["arm_fraction"]
        modes = torch.randint(1, 7, (count,), device=self.device)
        modes[torch.rand(count, device=self.device) < 0.25] = 0
        offset = arm_pose_bank(self.device)[modes] * level
        # Vary amplitude within each non-Home pose class, retaining endpoint cases.
        amplitude = 0.35 + 0.65 * torch.rand(count, 1, device=self.device)
        amplitude[torch.rand(count, device=self.device) < 0.25] = 1.0
        offset *= amplitude
        if self._env.cfg.week7_training_round >= 3 and self.override is None:
            from .symmetry import mirror_arm_offsets

            # Include the reflected asymmetric pose in real physics rollouts.
            reflected = torch.rand(count, 1, device=self.device) < 0.5
            offset = torch.where(reflected, mirror_arm_offsets(offset), offset)
        if self.override is not None:
            override = torch.as_tensor(
                self.override, dtype=torch.float32, device=self.device
            )
            offset = (
                override[env_ids]
                if override.ndim == 2 and override.shape[0] == self.num_envs
                else override.expand(count, 10)
            ).clone()
            modes[:] = -1
        home = self.robot.data.default_joint_pos[env_ids][:, self.ids]
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.ids]
        desired = (home + offset).clamp(min=limits[..., 0], max=limits[..., 1])
        self.offsets[env_ids] = desired - home
        self.targets[env_ids] = desired
        self.pose_index[env_ids] = modes
        # Spawn in the sampled held pose. No in-episode teleportation is used.
        self.robot.write_joint_state_to_sim(
            desired, torch.zeros_like(desired), self.ids, env_ids
        )

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


def arm_reference(env):
    return env.command_manager.get_command("arms")


@dataclass(kw_only=True)
class HeldArmActionCfg(ResidualActionCfg):
    def build(self, env):
        return HeldArmAction(self, env)


class HeldArmAction(ResidualAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.leg_columns = torch.tensor(
            [i for i, n in enumerate(self.names) if n not in ARM_NAMES],
            device=self.device,
        )
        self.leg_ids = self.ids[self.leg_columns]

    def process_actions(self, actions):
        super().process_actions(actions)
        arms = self._env.command_manager.get_term("arms")
        # Keep the 22-output history convention, but only 12 leg outputs actuate.
        # The ten upper-body outputs are superseded by the commanded references.
        self.targets[:, arms.ids] = arms.targets


@dataclass(kw_only=True)
class Week7TwistCfg(TwistCfg):
    def build(self, env):
        return Week7Twist(self, env)


class Week7Twist(Twist):
    def reset(self, env_ids):
        result = super().reset(env_ids)
        if self._env.cfg.week7_randomize_phase:
            self.gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        return result

    def _resample_command(self, env_ids):
        cap = env_targets(self._env)["forward_cap_m_s"]
        count = len(env_ids)
        draw = torch.rand(count, device=self.device)
        self.target[env_ids] = 0.0
        forward = torch.empty(count, device=self.device).uniform_(0.03, cap)
        forward[torch.rand(count, device=self.device) < 0.25] = cap
        reverse = -torch.empty(count, device=self.device).uniform_(0.03, 0.06)
        speed = torch.where((draw >= 0.70) & (draw < 0.85), reverse, forward)
        self.target[env_ids, 0] = torch.where(
            (draw >= 0.20) & (draw < 0.85), speed, 0.0
        )
        yaw = torch.empty(count, device=self.device).uniform_(0.08, 0.25)
        yaw *= torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        turning = (draw >= 0.85) | ((draw >= 0.55) & (draw < 0.70))
        self.target[env_ids, 2] = torch.where(turning, yaw, 0.0)
        anchor = (draw >= 0.20) & (draw < 0.35)
        self.target[env_ids[anchor], 0] = 0.06

    def _update_command(self):
        before = self.gait_phase.clone()
        super()._update_command()
        if self.command_dt == 0:
            return
        # Preserve R25 cadence up to 0.15 m/s, then shorten the cycle to 0.85 s.
        from sunday_locomotion.cadence import torch_period

        faster = ((self.value[:, 0] - 0.15) / 0.05).clamp(0, 1)
        period = torch_period(self.value) - 0.15 * faster
        self.gait_period[:] = period
        self.gait_phase[:] = (before + self._env.step_dt / period) % 1.0


def swing_targets(env):
    stance, u, _, _, old_z, _, _ = foot_targets(env)
    peak = env_targets(env)["swing_target_m"]
    return stance, u, old_z * (peak / 0.024), peak


def clearance_reward(env):
    _, _, target_z, _ = swing_targets(env)
    _, lowest = base.sole_geometry(env)
    quality = (1 - (lowest - target_z).abs() / 0.025).clamp(-1, 1)
    # A good stance foot cannot hide poor tracking of the other, swinging foot.
    reduced = (
        quality.amin(dim=1)
        if env.cfg.week7_training_round >= 2
        else quality.mean(dim=1)
    )
    return reduced * gait_weight(env)


def clearance_margin_cost(env):
    stance, u, _, peak = swing_targets(env)
    _, lowest = base.sole_geometry(env)
    middle = (~stance) & (u > 0.30) & (u < 0.70)
    if env.cfg.week7_training_round >= 4:
        # Track the actual 45 mm trajectory; avoid the old 75%-peak plateau.
        _, _, desired, _ = swing_targets(env)
        middle = (~stance) & (u > 0.20) & (u < 0.80)
        deficit = ((desired - lowest) / 0.010).clamp(min=0, max=5)
        return (deficit.square() * middle).amax(dim=1) * gait_weight(env)
    desired = 0.75 * peak * gait_weight(env)[:, None]
    deficit = ((desired - lowest) / 0.012).clamp(min=0, max=5)
    cost = deficit.square() * middle
    reduced = (
        cost.amax(dim=1) if env.cfg.week7_training_round >= 2 else cost.mean(dim=1)
    )
    return reduced * gait_weight(env)


def leg_posture(env, stopped=False):
    robot = env.scene["robot"].data
    ids = env.action_manager.get_term("residual").leg_ids
    error = (robot.joint_pos[:, ids] - robot.default_joint_pos[:, ids]) / (
        0.18 if stopped else 0.35
    )
    value = torch.exp(-error.square().mean(dim=1))
    if stopped:
        value *= env.command_manager.get_command("twist").abs().sum(dim=1) < 0.003
    return value


def leg_action_rate(env):
    ids = env.action_manager.get_term("residual").leg_columns
    return (
        (env.action_manager.action[:, ids] - env.action_manager.prev_action[:, ids])
        .square()
        .mean(dim=1)
    )


def arm_tracking(env):
    arms = env.command_manager.get_term("arms")
    error = (env.scene["robot"].data.joint_pos[:, arms.ids] - arms.targets) / 0.10
    return torch.exp(-error.square().mean(dim=1))


def foot_heading_error(env):
    """Signed L/R toe yaw relative to root and the commanded turning path."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    robot = env.scene["robot"]
    data = robot.data
    ids = robot.find_sites(("l_sole", "r_sole"), preserve_order=True)[0]
    rotation = matrix_from_quat(data.site_quat_w[:, ids])
    root = matrix_from_quat(data.root_link_quat_w)
    # Model toe direction is sole-local -Y, matching the archived heading reward.
    toe = torch.atan2(-rotation[..., 1, 1], -rotation[..., 0, 1])
    heading = torch.atan2(root[:, 1, 0], root[:, 0, 0])[:, None]
    desired = foot_targets(env)[-1]
    angle = toe - heading - desired
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def inward_angles(error):
    """Positive inward deviation: left toes right, right toes left (L/R order)."""
    signs = error.new_tensor([-1.0, 1.0])
    return (error * signs).clamp(min=0)


def foot_heading_cost(env):
    # Continuous cost avoids the near-zero gradient of exp at large toe-in angles.
    return (foot_heading_error(env) / 0.20).square().clamp(max=6).mean(dim=1)


def toe_in_cost(env):
    inward = (inward_angles(foot_heading_error(env)) - 0.035).clamp(min=0)
    return (inward / 0.15).square().clamp(max=6).amax(dim=1)


def lift_gated_base_reward(env, kind):
    """Use the current lift target in inherited speed/progress reward gating."""
    from sunday_locomotion.quality_mdp import motion_factor

    value = base.reward(env, kind)
    old = motion_factor(env)
    stance, u, target, _ = swing_targets(env)
    _, height = base.sole_geometry(env)
    data = env.scene["robot"].data
    ids = env.scene["robot"].find_sites(("l_sole", "r_sole"), preserve_order=True)[0]
    speed = data.site_lin_vel_w[:, ids, :2].norm(dim=-1)
    supported = base.foot_forces(env).norm(dim=-1) > 0.1 * env.cfg.sunday_mass * 9.81
    planted = torch.exp(-(speed / 0.04).square()) * supported
    planted = torch.where(stance, planted, 1.0).amin(dim=1)
    ratio = (height.clamp(min=0) / target.clamp(min=0.006)).clamp(0, 1)
    ratio = torch.where((~stance) & (u > 0.12) & (u < 0.88), ratio, 1.0).amin(dim=1)
    factor = 0.15 + 0.85 * planted * ratio
    factor = torch.where(gait_weight(env) > 0.08, factor, torch.ones_like(factor))
    return value * factor / old.clamp(min=0.15)


def moving_vertical_torso_cost(env):
    from mjlab.utils.lab_api.math import matrix_from_quat
    from .balance import torso_motion_cost_values

    data = env.scene["robot"].data
    rotation = matrix_from_quat(data.root_link_quat_w)
    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    height_error = (
        data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - env.cfg.sunday_h0
    ).abs()
    return torso_motion_cost_values(roll, height_error, gait_weight(env) > 0.9)
