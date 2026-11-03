"""Optional RSL-RL extensions absent from the frozen base config dataclass."""
from dataclasses import dataclass
from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class MirrorPpoCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict | None = None
