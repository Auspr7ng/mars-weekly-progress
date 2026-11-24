"""Compare opposite feet at corresponding swing phases, never simultaneous heights."""

import math
import torch
from mjlab.managers.manager_base import ManagerTermBase
from sunday_locomotion import mdp as base
from sunday_locomotion.control_mdp import foot_targets, gait_weight
from .curriculum import progress
from .symmetry import mirror_arm_offsets


class PhasePairMemory:
    def __init__(self, count, device, bins=8, max_age=200, only_raise=False):
        self.bins, self.max_age = bins, max_age
        self.only_raise = only_raise
        self.height = torch.zeros(count, 2, bins, device=device)
        self.seen = torch.zeros(count, 2, bins, dtype=torch.bool, device=device)
        self.age = torch.zeros(count, 2, bins, dtype=torch.long, device=device)
        self.rows = torch.arange(count, device=device)[:, None].expand(-1, 2)
        self.feet = torch.arange(2, device=device)[None, :].expand(count, -1)
        self.previous_command = torch.zeros(count, 3, device=device)

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self.height[selected] = 0
        self.seen[selected] = False
        self.age[selected] = 0
        self.previous_command[selected] = 0

    def update(self, heights, swing_phase, swinging, eligible, command):
        # Do not pair samples collected at different commanded speeds/turn rates.
        changed = (command - self.previous_command).abs().amax(dim=1) > 0.005
        clear = ~eligible | changed
        self.seen[clear] = False
        self.previous_command.copy_(command)
        self.age += 1
        indices = (swing_phase * self.bins).long().clamp(0, self.bins - 1)
        active = (
            swinging & (swing_phase > 0.15) & (swing_phase < 0.85) & eligible[:, None]
        )
        opposite = 1 - self.feet
        reference = self.height[self.rows, opposite, indices]
        valid = active & self.seen[self.rows, opposite, indices]
        valid &= self.age[self.rows, opposite, indices] < self.max_age
        # Each sample is compared with a corresponding bin of the previous
        # opposite-foot swing. Both feet also retain their absolute lift target.
        difference = (
            (reference - heights - 0.002).clamp(min=0)
            if self.only_raise
            else (heights - reference)
        )
        error = (difference / 0.006).square().clamp(max=9)
        cost = (error * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        old = self.height[self.rows, self.feet, indices]
        seen = self.seen[self.rows, self.feet, indices]
        updated = torch.where(seen, 0.5 * old + 0.5 * heights, heights)
        self.height[self.rows, self.feet, indices] = torch.where(active, updated, old)
        self.seen[self.rows, self.feet, indices] |= active
        age = self.age[self.rows, self.feet, indices]
        self.age[self.rows, self.feet, indices] = torch.where(active, 0, age)
        return cost, valid.any(dim=1)


def symmetric_straight_gate(env):
    command = env.command_manager.get_command("twist")
    arms = env.command_manager.get_command("arms")
    symmetric = (arms - mirror_arm_offsets(arms)).abs().amax(dim=1) < 0.015
    straight = (command[:, 0].abs() > 0.025) & (command[:, 1:].abs().amax(dim=1) < 0.01)
    ready = (env.episode_length_buf * env.step_dt > 1.4) & (gait_weight(env) > 0.9)
    return symmetric & straight & ready


class PhaseMatchedClearanceCost(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.memory = PhasePairMemory(
            env.num_envs, env.device, only_raise=env.cfg.week7_training_round >= 4
        )
        self.last_cost = torch.zeros(env.num_envs, device=env.device)
        self.last_paired = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None):
        self.memory.reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        self.last_cost[selected] = 0
        self.last_paired[selected] = False

    def __call__(self, env):
        stance, phase, *_ = foot_targets(env)
        _, height = base.sole_geometry(env)
        self.last_cost, self.last_paired = self.memory.update(
            height.clamp(min=0),
            phase,
            ~stance,
            symmetric_straight_gate(env),
            env.command_manager.get_command("twist"),
        )
        # Ease in the new regularizer without lowering the absolute 45 mm target.
        return self.last_cost * (0.25 + 0.75 * progress(env))


class LateralBiasCost(ManagerTermBase):
    """Penalize sustained excess lateral velocity while allowing planned sway."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self.bias = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.bias[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        from sunday_locomotion.quality_mdp import lateral_velocity_target

        command = env.command_manager.get_command("twist")
        eligible = (
            (command[:, 0].abs() > 0.025)
            & (command[:, 2].abs() < 0.01)
            & (gait_weight(env) > 0.9)
        )
        excess = (
            env.scene["robot"].data.root_link_lin_vel_b[:, 1]
            - command[:, 1]
            - lateral_velocity_target(env)
        )
        alpha = 1 - math.exp(-env.step_dt / 0.5)
        self.bias = torch.where(eligible, self.bias + alpha * (excess - self.bias), 0.0)
        return (self.bias / 0.015).square().clamp(max=9) * eligible
