#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${MJLAB_PYTHON:-python3}" code/run_demo.py "$@"
