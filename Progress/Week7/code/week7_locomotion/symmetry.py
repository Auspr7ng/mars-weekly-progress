"""Soft mirror prior for the 510/108 held-arm schema; never fake PPO samples."""

import torch
from sunday_locomotion import symmetry as archived
from .curriculum import ARM_NAMES


def mirror_arm_offsets(offsets):
    """Home arm references are mirror symmetric; offsets are R5 then L5."""
    indices = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], device=offsets.device)
    signs = offsets.new_tensor([-1, 1, -1, 1, -1] * 2)
    return offsets[..., indices] * signs


def reflect_group(value, group, mapping):
    prefix, history = {"actor": (460, 5), "critic": (98, 1)}[group]
    if value.shape[-1] != prefix + history * 10:
        raise ValueError(f"Unexpected Week7 {group} reflection schema")
    result = value.clone()
    result[..., :prefix] = archived.reflect_group(value[..., :prefix], group, mapping)
    refs = value[..., prefix:].reshape(*value.shape[:-1], history, 10)
    result[..., prefix:] = mirror_arm_offsets(refs).flatten(-2)
    return result


def augment(env, obs=None, actions=None):
    base = env.unwrapped
    mapping = archived.mapping(env)
    # The inherited affine map includes non-mirrored nominal leg offsets.
    # Verify this version's zero-offset arm-reference reflection assumption.
    if not getattr(base, "_week7_arm_mirror_checked", False):
        robot = base.scene["robot"]
        ids = [robot.joint_names.index(n) for n in ARM_NAMES]
        if not torch.allclose(
            mapping["offset"][ids], torch.zeros(10, device=base.device), atol=1e-6
        ):
            raise ValueError("Arm Home is no longer mirror symmetric")
        base._week7_arm_mirror_checked = True
    if obs is not None:
        reflected = obs.clone()
        for group in obs.keys():
            reflected[group] = reflect_group(obs[group], group, mapping)
        obs = torch.cat([obs, reflected], dim=0)
    if actions is not None:
        actions = torch.cat(
            [actions, archived.reflect_actions(actions, mapping)], dim=0
        )
    return obs, actions


class LegOnlyMirrorLoss:
    """RSL-RL-compatible mirror extension restricted to actuated leg outputs."""

    @staticmethod
    def install(runner):
        from rsl_rl.extensions.symmetry import Symmetry

        class LegSymmetry(Symmetry):
            def compute_loss(self, actor, batch, original_batch_size):
                if self.use_data_augmentation:
                    raise ValueError(
                        "Leg mirror prior does not authorize synthetic PPO data"
                    )
                batch.observations, _ = self.data_augmentation_func(
                    env=self.env, obs=batch.observations, actions=None
                )
                means = actor(batch.observations.detach().clone())
                _, reflected = self.data_augmentation_func(
                    env=self.env, obs=None, actions=means[:original_batch_size]
                )
                legs = self.env.unwrapped.action_manager.get_term(
                    "residual"
                ).leg_columns
                loss = (
                    (
                        means[original_batch_size:, legs]
                        - reflected.detach()[original_batch_size:, legs]
                    )
                    .square()
                    .mean()
                )
                return loss if self.use_mirror_loss else loss.detach()

        old = runner.alg.symmetry
        runner.alg.symmetry = LegSymmetry(
            env=old.env,
            data_augmentation_func=old.data_augmentation_func,
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=old.mirror_loss_coeff,
        )
