"""Four sole-mounted 32x32 arrays: toe and heel per foot, Week1-style SDF."""
from dataclasses import dataclass

import mujoco
import numpy as np

from .flexitac import FourFingerFlexiTac

FOOT_ROWS = FOOT_COLS = 32
FOOT_PITCH_M = .0025
FOOT_SIDE_M = .080
SOLE_Z = -.030
STANDING_LOWER_M = .006
FOOT_RESPONSE_DEPTH_M = .002


@dataclass(frozen=True)
class FootPatch:
    name: str
    link: str
    center: tuple
    color: tuple
    opposite: None = None


# Current Oct21 CAD: sole X width 105 mm, Y length 180 mm; toe = -Y.
# Each 80 mm square is inset 2 mm from its end; center gap is 16 mm.
FOOT_PATCHES = (
    FootPatch("left_front", "l_foot_link_1", (.005, -.099175, SOLE_Z), (.2, .9, .4)),
    FootPatch("left_heel", "l_foot_link_1", (.005, -.003175, SOLE_Z), (.1, .7, 1.)),
    FootPatch("right_front", "r_foot_link_1", (-.005, -.099175, SOLE_Z), (1., .65, .1)),
    FootPatch("right_heel", "r_foot_link_1", (-.005, -.003175, SOLE_Z), (.8, .3, .9)),
)


def foot_local_taxels():
    # Image rows run from toe (-Y) to heel (+Y); columns run along local +X.
    x, y = np.meshgrid((np.arange(FOOT_COLS) - 15.5) * FOOT_PITCH_M,
                       (np.arange(FOOT_ROWS) - 15.5) * FOOT_PITCH_M)
    offsets = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=-1)
    return np.asarray([offsets + f.center for f in FOOT_PATCHES], dtype=np.float32)


def attach_foot_sites(spec, prefix=""):
    for patch in FOOT_PATCHES:
        name = prefix + "flexitac_" + patch.name
        if any(site.name == name for site in spec.sites):
            continue
        spec.body(prefix + patch.link).add_site(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=patch.center, size=(FOOT_SIDE_M / 2, FOOT_SIDE_M / 2, .00015),
            rgba=(*patch.color, .45), group=3)


class FootTactileEngine(FourFingerFlexiTac):
    def __init__(self, model, num_envs=1, device="cuda:0", prefix=""):
        super().__init__(model, num_envs, device, prefix,
                         patches=FOOT_PATCHES, points_local=foot_local_taxels(),
                         grid_shape=(FOOT_ROWS, FOOT_COLS), shell_m=0.,
                         response_depth_m=FOOT_RESPONSE_DEPTH_M)
        self.result.active_threshold = 1.e-6
