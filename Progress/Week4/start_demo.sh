#!/usr/bin/env bash
set -eu
week_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONUTF8=1
exec "${WEEK4_PYTHON:-python}" -B -u "$week_dir/code/run_demo.py" "$@"
