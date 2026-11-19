"""Safety-critical Xbox mapping tests without opening hardware."""

import unittest

from jetson_xbox_shadow import Deadman, ZERO, map_left_stick, normalize


class XboxShadowInputTest(unittest.TestCase):
    def test_axis_normalization(self):
        self.assertEqual(normalize(0, 0, 65535), -1.0)
        self.assertEqual(normalize(65535, 0, 65535), 1.0)
        self.assertAlmostEqual(normalize(32768, 0, 65535), 0.0, places=4)

    def test_mapping_limits_and_signs(self):
        self.assertEqual(map_left_stick(0, -1), (0.06, 0.0, 0.0))
        self.assertEqual(map_left_stick(0, 1), (-0.04, 0.0, 0.0))
        self.assertEqual(map_left_stick(-1, 0), (0.0, 0.0, 0.15))
        self.assertEqual(map_left_stick(1, 0), (0.0, 0.0, -0.15))
        self.assertEqual(map_left_stick(0.1, -0.1), ZERO)

    def test_neutral_then_lb_is_required(self):
        d = Deadman()
        self.assertEqual(d.update(connected=True, y=-1, lb=True), ZERO)
        self.assertEqual(d.update(connected=True, y=0, lb=False), ZERO)
        self.assertEqual(d.update(connected=True, y=-1, lb=True), (0.06, 0.0, 0.0))
        self.assertEqual(d.update(connected=True, y=-1, lb=False), ZERO)
        self.assertEqual(d.update(connected=True, y=-1, lb=True), ZERO)

    def test_b_stop_latches_until_rearmed(self):
        d = Deadman()
        d.update(connected=True)
        d.update(connected=True, y=-1, lb=True)
        self.assertEqual(d.update(connected=True, y=-1, lb=True, b=True), ZERO)
        self.assertEqual(d.update(connected=True, y=-1, lb=True), ZERO)
        d.update(connected=True, lb=False)
        self.assertEqual(d.update(connected=True, y=-1, lb=True), (0.06, 0.0, 0.0))

    def test_disconnect_requires_rearm(self):
        d = Deadman()
        d.update(connected=True)
        d.update(connected=True, y=-1, lb=True)
        self.assertEqual(d.update(connected=False), ZERO)
        self.assertEqual(d.update(connected=True, y=-1, lb=True), ZERO)


if __name__ == "__main__":
    unittest.main()
