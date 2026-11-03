"""Use the frozen Week4 framework and write all runtime caches inside Week4."""
import os
from pathlib import Path
import sys

WEEK = Path(__file__).resolve().parents[2]
for entry in (WEEK / "code/vendor", WEEK / "code/src"):
    sys.path.insert(0, str(entry))
for name, subdir in {
    "MPLCONFIGDIR": "matplotlib", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions", "XDG_CACHE_HOME": "xdg",
}.items():
    os.environ.setdefault(name, str(WEEK / ".cache" / subdir))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")


def configure_warp():
    import warp as wp
    wp.config.kernel_cache_dir = str(WEEK / ".cache/warp")
