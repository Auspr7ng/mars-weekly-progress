"""Regression checks for raise-only pairing and leg-only mirror optimization."""

import bootstrap
from types import SimpleNamespace
import unittest
import torch
from week7_locomotion.balance import PhasePairMemory
from week7_locomotion.symmetry import LegOnlyMirrorLoss
from week7_locomotion.curriculum import targets


class Round4Tests(unittest.TestCase):
    def test_higher_foot_is_not_penalized_for_exceeding_the_other(self):
        memory = PhasePairMemory(1, "cpu", only_raise=True)
        command = torch.tensor([[0.12, 0.0, 0.0]])
        phase = torch.full((1, 2), 0.5)
        eligible = torch.tensor([True])
        memory.update(
            torch.tensor([[0.025, 0.0]]),
            phase,
            torch.tensor([[True, False]]),
            eligible,
            command,
        )
        cost, paired = memory.update(
            torch.tensor([[0.0, 0.04]]),
            phase,
            torch.tensor([[False, True]]),
            eligible,
            command,
        )
        self.assertTrue(paired.item())
        self.assertEqual(cost.item(), 0.0)
        cost, paired = memory.update(
            torch.tensor([[0.025, 0.0]]),
            phase,
            torch.tensor([[True, False]]),
            eligible,
            command,
        )
        self.assertTrue(paired.item())
        self.assertGreater(cost.item(), 1.0)

    def test_mirror_loss_ignores_all_ten_nonactuating_arm_outputs(self):
        def augment(env, obs=None, actions=None):
            return (
                torch.cat([obs, obs]) if obs is not None else None,
                torch.cat([actions, actions]) if actions is not None else None,
            )

        env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                action_manager=SimpleNamespace(
                    get_term=lambda name: SimpleNamespace(
                        leg_columns=torch.arange(10, 22)
                    )
                )
            )
        )
        runner = SimpleNamespace(
            alg=SimpleNamespace(
                symmetry=SimpleNamespace(
                    env=env, data_augmentation_func=augment, mirror_loss_coeff=0.05
                )
            )
        )
        LegOnlyMirrorLoss.install(runner)

        def actor(obs):
            means = torch.zeros(len(obs), 22)
            means[2:, :10] = 100.0
            return means

        loss = runner.alg.symmetry.compute_loss(
            actor, SimpleNamespace(observations=torch.zeros(2, 1)), 2
        )
        self.assertEqual(loss.item(), 0.0)

        def bad_leg_actor(obs):
            means = actor(obs)
            means[2:, 10] = 1.0
            return means

        loss = runner.alg.symmetry.compute_loss(
            bad_leg_actor, SimpleNamespace(observations=torch.zeros(2, 1)), 2
        )
        self.assertAlmostEqual(loss.item(), 1 / 12, places=6)

    def test_round4_preserves_absolute_objectives(self):
        for level in (0.0, 0.5, 1.0):
            self.assertEqual(targets(level, 4)["swing_target_m"], 0.045)
            self.assertEqual(targets(level, 4)["forward_cap_m_s"], 0.20)


if __name__ == "__main__":
    unittest.main()
