"""Print the latest persisted training report without loading CUDA or hardware."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
print(
    json.dumps(
        json.loads((args.run_dir / "status.json").read_text(encoding="utf-8")), indent=2
    )
)
