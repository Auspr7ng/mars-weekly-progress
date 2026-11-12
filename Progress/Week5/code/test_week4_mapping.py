"""Verify the physical left stick retains Week4's body-frame command contract."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'Week4/code/src'))
from controller import LIMITS, axis
from week4_mapping import map_left_stick
from sunday_locomotion.arrow_control import joystick_to_command, MAX_LENGTH


class Week4Parity(unittest.TestCase):
    def test_original_arrow_parity(self):
        for x, y in ((0,0), (0,-1), (0,1), (-1,0), (1,0), (.7,-.7), (-1,1)):
            vector = np.array([-axis(y), -axis(x), 0.])*MAX_LENGTH
            norm = np.linalg.norm(vector)
            if norm > MAX_LENGTH:
                vector *= MAX_LENGTH/norm
            expected, _ = joystick_to_command(vector, True, **LIMITS)
            np.testing.assert_allclose(map_left_stick(x,y,LIMITS), expected)

    def test_diagonal_walk_and_turn(self):
        command = map_left_stick(-.7,-.7,LIMITS)
        self.assertGreater(command[0], 0.)
        self.assertEqual(command[1], 0.)
        self.assertGreater(command[2], 0.)

    def test_centre_and_drift_stop(self):
        self.assertEqual(map_left_stick(.03,.06,LIMITS), [0.,0.,0.])


if __name__ == '__main__':
    unittest.main()
