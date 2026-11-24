"""Use the frozen Week4 runtime and keep new artifacts inside Week7."""

import os
from pathlib import Path
import sys

WEEK = Path(__file__).resolve().parents[1]
BASE = WEEK.parent / "Week4"
sys.path[1:1] = [str(WEEK / "code"), str(BASE / "code/vendor"), str(BASE / "code/src")]
for name, directory in {
    "MPLCONFIGDIR": "matplotlib",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "XDG_CACHE_HOME": "xdg",
}.items():
    os.environ.setdefault(name, str(WEEK / ".cache" / directory))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")


def configure_warp():
    import warp as wp

    wp.config.kernel_cache_dir = str(WEEK / ".cache/warp")
