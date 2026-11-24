"""Tests for curriculum boundaries and exact transfer invariants."""

import bootstrap
import math
import unittest

import torch
from week7_locomotion.curriculum import targets, ARM_NAMES
from week7_locomotion.mdp import arm_pose_bank, inward_angles
from week7_locomotion.transfer import expand_state


class ContractTests(unittest.TestCase):
    def test_targets_and_endpoints(self):
        first, last = targets(0), targets(1)
        self.assertAlmostEqual(first["forward_cap_m_s"], 0.12)
        self.assertAlmostEqual(last["forward_cap_m_s"], 0.20)
        self.assertAlmostEqual(last["swing_target_m"], 0.035)
        self.assertEqual(first["reverse_cap_m_s"], last["reverse_cap_m_s"])
        self.assertEqual(targets(-1), first)
        self.assertEqual(targets(2), last)
        with self.assertRaises(ValueError):
            targets(float("nan"))

    def test_round2_targets_and_toe_in_signs(self):
        self.assertAlmostEqual(targets(0, 2)["swing_target_m"], 0.035)
        self.assertAlmostEqual(targets(1, 2)["swing_target_m"], 0.045)
        self.assertAlmostEqual(targets(1, 2)["forward_cap_m_s"], 0.20)
        self.assertEqual(targets(0, 2)["arm_fraction"], 1.0)
        errors = torch.tensor([[-0.2, 0.3], [0.2, -0.3], [0.0, 0.0]])
        torch.testing.assert_close(
            inward_angles(errors), torch.tensor([[0.2, 0.3], [0.0, 0.0], [0.0, 0.0]])
        )

    def test_round3_keeps_absolute_lift_and_speed_targets(self):
        for level in (0, 0.5, 1):
            self.assertAlmostEqual(targets(level, 3)["swing_target_m"], 0.045)
            self.assertAlmostEqual(targets(level, 3)["forward_cap_m_s"], 0.20)
            self.assertEqual(targets(level, 3)["arm_fraction"], 1.0)

    def test_pose_directions_and_asymmetry(self):
        bank = arm_pose_bank()
        self.assertEqual(len(ARM_NAMES), 10)
        self.assertEqual(bank.shape, (7, 10))
        self.assertTrue(torch.equal(bank[0], torch.zeros(10)))
        self.assertGreater(bank[1, ARM_NAMES.index("r_shoulder_roll")], 0)
        self.assertLess(bank[1, ARM_NAMES.index("l_shoulder_roll")], 0)
        self.assertGreater(bank[2, ARM_NAMES.index("r_shoulder_pitch")], 0)
        self.assertGreater(bank[2, ARM_NAMES.index("l_shoulder_pitch")], 0)
        self.assertNotEqual(abs(float(bank[6, 0])), abs(float(bank[6, 5])))

    def test_expanded_inputs_preserve_old_network_for_any_new_reference(self):
        torch.manual_seed(7)
        old = {
            "mlp.0.weight": torch.randn(8, 4),
            "mlp.0.bias": torch.randn(8),
            "obs_normalizer._mean": torch.randn(1, 4),
            "obs_normalizer._std": torch.rand(1, 4) + 0.1,
            "obs_normalizer._var": torch.ones(1, 4),
            "obs_normalizer.count": torch.tensor(500_000),
        }
        new = {
            **old,
            "mlp.0.weight": torch.zeros(8, 7),
            "obs_normalizer._mean": torch.zeros(1, 7),
            "obs_normalizer._std": torch.ones(1, 7),
            "obs_normalizer._var": torch.ones(1, 7),
        }
        state = expand_state(old, new)
        x = torch.randn(20, 4)
        y = torch.cat((x, torch.randn(20, 3) * 20), dim=1)
        expected = (x - old["obs_normalizer._mean"]) / (
            old["obs_normalizer._std"] + 0.01
        ) @ old["mlp.0.weight"].T + old["mlp.0.bias"]
        actual = (y - state["obs_normalizer._mean"]) / (
            state["obs_normalizer._std"] + 0.01
        ) @ state["mlp.0.weight"].T + state["mlp.0.bias"]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(int(state["obs_normalizer.count"]), 200_000)
        self.assertEqual(int(old["obs_normalizer.count"]), 500_000)
        self.assertEqual(torch.count_nonzero(state["mlp.0.weight"][:, 4:]), 0)


if __name__ == "__main__":
    unittest.main()
