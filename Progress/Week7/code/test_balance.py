"""Phase-pair regression checks, including reset and transition isolation."""

import bootstrap
import unittest
import torch
from week7_locomotion.balance import PhasePairMemory
from week7_locomotion.symmetry import mirror_arm_offsets


class BalanceTests(unittest.TestCase):
    def setUp(self):
        self.memory = PhasePairMemory(2, "cpu")
        self.command = torch.tensor([[0.12, 0.0, 0.0], [0.12, 0.0, 0.0]])
        self.eligible = torch.tensor([True, True])

    def sample(self, left_height, right_height, left_swing, right_swing, phase=0.5):
        return self.memory.update(
            torch.tensor([[left_height, right_height]] * 2),
            torch.full((2, 2), phase),
            torch.tensor([[left_swing, right_swing]] * 2),
            self.eligible,
            self.command,
        )

    def test_equal_opposite_swings_pass_despite_unequal_instant_heights(self):
        _, paired = self.sample(0.04, 0.0, True, False)
        self.assertFalse(paired.any())
        cost, paired = self.sample(0.0, 0.04, False, True)
        self.assertTrue(paired.all())
        torch.testing.assert_close(cost, torch.zeros(2))

    def test_low_opposite_swing_is_penalized(self):
        self.sample(0.04, 0.0, True, False)
        cost, paired = self.sample(0.0, 0.025, False, True)
        self.assertTrue(paired.all())
        self.assertTrue((cost > 1).all())

    def test_different_phases_and_ineligible_samples_are_not_paired(self):
        self.sample(0.04, 0.0, True, False, phase=0.2)
        _, paired = self.sample(0.0, 0.04, False, True, phase=0.8)
        self.assertFalse(paired.any())
        self.eligible[:] = False
        cost, paired = self.sample(0.04, 0.01, True, False)
        self.assertFalse(paired.any())
        self.assertTrue((cost == 0).all())

    def test_subset_reset_and_command_change_clear_history(self):
        self.sample(0.04, 0.0, True, False)
        self.memory.reset(torch.tensor([1]))
        _, paired = self.sample(0.0, 0.04, False, True)
        self.assertEqual(paired.tolist(), [True, False])
        self.command[:, 0] = 0.20
        _, paired = self.sample(0.04, 0.0, True, False)
        self.assertFalse(paired.any())

    def test_asymmetric_arm_reference_reflects_to_its_counterpart(self):
        value = torch.tensor([[0.15, 0.35, 0.0, 0.0, 0.0, -0.50, 0.10, 0.0, 0.0, 0.0]])
        expected = torch.tensor(
            [[0.50, 0.10, 0.0, 0.0, 0.0, -0.15, 0.35, 0.0, 0.0, 0.0]]
        )
        torch.testing.assert_close(mirror_arm_offsets(value), expected)
        torch.testing.assert_close(mirror_arm_offsets(expected), value)


if __name__ == "__main__":
    unittest.main()
