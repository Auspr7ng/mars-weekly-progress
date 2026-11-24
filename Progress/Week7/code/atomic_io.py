"""Atomic local files with bounded retries for transient Windows reader locks."""

import json
import os
from pathlib import Path
import time


def replace_with_retry(source, destination, attempts=40, delay_s=0.05):
    """Keep the old destination intact until replacement succeeds, or raise."""
    if attempts < 1 or delay_s < 0:
        raise ValueError("Positive attempts and nonnegative retry delay are required")
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    replace_with_retry(temporary, path)
