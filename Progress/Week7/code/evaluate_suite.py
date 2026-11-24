"""Run the bounded Week7 independent evaluation suite for one checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

WEEK = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prefix", required=True, help="Unique result filename prefix")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("prefix may contain only letters, digits, underscores and hyphens")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or not (checkpoint.parent / "run.json").is_file():
        parser.error("checkpoint and adjacent run.json are required")
    results, logs = WEEK / "results", WEEK / ".cache/suites"
    results.mkdir(exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("short", "evaluate.py", ["--trials", "4", "--seconds", "6", "--seed", "1742"]),
        (
            "extended",
            "evaluate.py",
            ["--trials", "8", "--seconds", "12", "--seed", "2742"],
        ),
        (
            "random_phase",
            "evaluate.py",
            ["--trials", "4", "--seconds", "12", "--seed", "3742", "--random-phase"],
        ),
        (
            "holdout20",
            "evaluate.py",
            ["--trials", "4", "--seconds", "20", "--seed", "6742"],
        ),
        ("native", "check_native.py", ["--seconds", "8"]),
        ("transitions", "check_transitions.py", ["--trials", "4", "--seed", "5742"]),
    ]
    if args.video:
        jobs.append(("preview", "record_preview.py", ["--seconds", "6"]))
    report = dict(
        schema="sunday.week7.evaluation_suite.v1",
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        results={},
        state="running",
        accepted=False,
    )
    manifest = results / (args.prefix + "_suite.json")
    if manifest.exists():
        raise ValueError("Use a new prefix; existing suite results are preserved")
    started = time.perf_counter()
    try:
        for name, script, options in jobs:
            output = results / (args.prefix + "_" + name + ".json")
            if name == "preview":
                output = results / "videos" / (args.prefix + "_preview.mp4")
            if output.exists():
                raise ValueError(f"Existing result is not overwritten: {output}")
            command = [
                sys.executable,
                "-B",
                "-u",
                str(WEEK / "code" / script),
                str(checkpoint),
                *options,
                "--output",
                str(output),
            ]
            with (logs / (args.prefix + "_" + name + ".log")).open(
                "w", encoding="utf-8"
            ) as stream:
                subprocess.run(
                    command,
                    cwd=WEEK,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report["results"][name] = str(output.relative_to(WEEK))
            report["elapsed_s"] = time.perf_counter() - started
            manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("SUITE_CASE_DONE " + name, flush=True)
        report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed", error=repr(exc))
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - started
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WEEK7_SUITE_COMPLETED " + str(manifest), flush=True)


if __name__ == "__main__":
    main()
