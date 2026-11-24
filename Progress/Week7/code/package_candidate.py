"""Package one measured experimental checkpoint and its immutable source snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

WEEK = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.name):
        parser.error(
            "name must contain lowercase letters, numbers, underscores or hyphens"
        )
    checkpoint, report = args.checkpoint.resolve(), args.report.resolve()
    if not checkpoint.is_file() or not report.is_file():
        parser.error("checkpoint and reviewed result report must exist")
    source = checkpoint.parent
    spec = json.loads((source / "run.json").read_text())
    if spec["actor_inputs"] != 510 or spec["actor_outputs"] != 22:
        raise ValueError("Expected the Week7 held-reference contract")
    hashes = {
        name.replace("\\", "/"): value for name, value in spec["source_hashes"].items()
    }
    for name, expected in hashes.items():
        if sha(source / "source" / name) != expected:
            raise ValueError(f"Training source hash mismatch: {name}")
    output = WEEK / "policies" / args.name
    if output.exists():
        raise ValueError("Existing policy bundles are preserved; choose a new name")
    output.mkdir(parents=True)
    shutil.copyfile(checkpoint, output / checkpoint.name)
    shutil.copyfile(source / "run.json", output / "run.json")
    with zipfile.ZipFile(
        output / "training_source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in sorted(hashes):
            archive.write(source / "source" / name, arcname=name)
    (output / "source_manifest.json").write_text(
        json.dumps(
            dict(files=hashes, snapshot_zip_sha256=sha(output / "training_source.zip")),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(WEEK / "code/export_actor.py"),
            str(output / checkpoint.name),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    metadata = dict(
        schema="sunday.week7.experimental_policy_package.v1",
        name=args.name,
        checkpoint=checkpoint.name,
        checkpoint_sha256=sha(output / checkpoint.name),
        actor_npz_sha256=sha(output / "actor.npz"),
        training_source_zip_sha256=sha(output / "training_source.zip"),
        training_round=spec["training_round"],
        reward_version=spec["reward_version"],
        source_run=source.name,
        source_iteration=int(checkpoint.stem.removeprefix("model_").split("_")[0]),
        report=str(report.relative_to(WEEK)),
        actor_inputs=510,
        critic_inputs=108,
        raw_outputs=22,
        applied_leg_outputs=12,
        accepted=False,
        hardware_release=False,
        current_imu="ST LSM6DSV16X",
        hardware_adapter_tested=False,
        provenance_note="run.json preserves original training paths; those historical absolute paths need not exist on another host",
    )
    (output / "package.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        f"""# {args.name} - Experimental simulation candidate

Checkpoint: `{checkpoint.name}`, Round {spec['training_round']}, source run
`{source.name}`. Read the [measured results](../../{report.relative_to(WEEK).as_posix()})
before choosing this candidate. It is not a hardware release and does not
replace the accepted R25 baseline.

## Files

- `{checkpoint.name}`: original training checkpoint, including critic/optimizer.
- `actor.npz`: deterministic NumPy actor and observation normalizer.
- `actor_contract.json`: input/action ownership, checksums and interface limits.
- `actor_parity.json`: numerical export comparison against checkpoint tensors.
- `run.json`: original training metadata and configuration/source hashes.
- `training_source.zip`, `source_manifest.json`: hash-verified training snapshot.
- `package.json`, `SHA256SUMS.txt`: package provenance and integrity metadata.

The source snapshot is the code present at training launch. Current evaluation
and packaging helpers may be newer and are versioned separately in Git.
Archived paths inside the historical run metadata need not exist locally.
Checksum-covered text files use LF line endings for Git portability; the
source archive preserves the original training-file bytes.

The actor has 510 inputs and retains 22 raw outputs/history values, but only
twelve leg outputs actuate. Ten arm targets come from held references.
Moon's 460-input R25 runner and historical BNO055 driver are incompatible
without adaptation. The current LSM6DSV16X wiring/mount/driver and body-speed
estimator have not been validated by this simulation package.
""",
        encoding="utf-8",
    )
    # Git stores these text types as LF. Hash those same bytes on Windows too.
    for path in output.iterdir():
        if path.suffix in (".json", ".md", ".txt"):
            path.write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS.txt").write_text(
        "".join(sha(path) + "  " + path.name + "\n" for path in files),
        encoding="utf-8",
        newline="\n",
    )
    print("WEEK7_PACKAGE_CREATED " + str(output), flush=True)


if __name__ == "__main__":
    main()
