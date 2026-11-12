"""Physical left stick deflects the same body-frame joystick used in Week4."""
from controller import axis
from sunday_locomotion.arrow_control import joystick_to_command, MAX_LENGTH
import numpy as np


def map_left_stick(x, y, limits):
    # Hardware dead zone removes measured stick drift. Then use the frozen
    # Week4 arrow's circular travel, per-axis response and minimum gait commands.
    vector = np.array([-axis(y), -axis(x), 0.])*MAX_LENGTH
    length = float(np.linalg.norm(vector))
    if length > MAX_LENGTH:
        vector *= MAX_LENGTH/length
    command, _ = joystick_to_command(vector, True, **limits)
    return command.tolist()
