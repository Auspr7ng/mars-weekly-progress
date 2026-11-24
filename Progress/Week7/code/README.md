# Week7 code and execution

Use the compatible Python 3.11/CUDA environment used for Week4. Keep Week4
and Week7 as siblings under `Progress/`; Week7 reuses the frozen Week4 mjlab
sources and model assets. Simulation scripts require Torch, MuJoCo, Warp and
the existing RL dependencies. No runtime package is installed automatically.

## Start and monitor

From Week7 on this workstation:

```powershell
$env:WEEK7_PYTHON = 'D:\robotics\SundayA1_mjlab\.venv\Scripts\python.exe'
.\start_round4.bat
.\open_training_monitor.bat
```

Set `WEEK7_PYTHON` to the compatible interpreter on another machine. Each
training launch requires a new run directory. The Round 4 launcher uses the
published Round 2 parent bundle, so ignored local historical runs are not
needed to reproduce its initialization. The launch is bounded to 800 updates
and 512 parallel environments. Earlier launchers describe historical local
experiments; their parent paths may require the corresponding local run.

The monitor reads files every two seconds without loading CUDA. It flags
stale running reports and queries Windows process presence. Reward and
average episode length are training indicators, not gait acceptance.

## Core files

| File | Purpose |
| --- | --- |
| `train.py` | Hash checks, source/configuration snapshots, atomic checkpoints, local status, bounded PPO |
| `week7_locomotion/curriculum.py` | Versioned 510/108/22 contract and target schedules |
| `week7_locomotion/mdp.py` | Held arms, twelve applied leg outputs, speed/cadence and target-aware lift shaping |
| `week7_locomotion/balance.py` | Corresponding-phase height history and sustained lateral-bias cost |
| `week7_locomotion/symmetry.py` | New-schema reflection and leg-only soft mirror loss |
| `week7_locomotion/transfer.py` | R25 weight/normalizer expansion into the new observation layout |
| `evaluate.py` | Independent first-episode pose/speed matrices and bilateral diagnostics |
| `check_native.py` | Native MuJoCo physics with CPU inference and initial observation parity |
| `record_preview.py` | Actual GPU-rollout video with full-body and foot views |
| `export_actor.py` | NumPy-only deterministic actor export and numerical parity checks |
| `training_monitor.py`, `status.py` | GUI and JSON persisted progress readers |

## Checks and evaluation

From Week7:

```powershell
& $env:WEEK7_PYTHON -B -m unittest discover -s code -p 'test_*.py' -v
& $env:WEEK7_PYTHON -B code/check_env.py --round 4 --output results/round4_environment_check.json
& $env:WEEK7_PYTHON -B code/evaluate.py runs/<run>/model_<iteration>.pt --trials 8 --seconds 12 --seed 2742 --output results/candidate_matrix.json
& $env:WEEK7_PYTHON -B code/evaluate.py runs/<run>/model_<iteration>.pt --trials 4 --seconds 12 --seed 3742 --random-phase --output results/candidate_random_phase.json
& $env:WEEK7_PYTHON -B code/check_native.py runs/<run>/model_<iteration>.pt --seconds 8 --output results/candidate_native.json
& $env:WEEK7_PYTHON -B code/record_preview.py runs/<run>/model_<iteration>.pt --seconds 6 --output results/videos/candidate_preview.mp4
```

The evaluator reads the checkpoint directory's `run.json`. Compare parent
and candidate using identical commands, duration, seed and phase setting.
Fixed initial phase is the historical default; `--random-phase` must be used
on both models for a matched random-phase comparison. Reports retain missing
post-settle sample counts. An early failed episode's zero-valued aggregate is
not evidence of good tracking.

The original walking gate requires survival, actual speed >=80% of command,
at least two completed swings per foot, both mean whole-sole swing peaks
>=20 mm and drift <0.15 m. The separate symmetric-straight diagnostic adds
both mean peaks >=30 mm, peak gap <=5 mm, inward toe yaw <=5 degrees each,
roll RMS <=3 degrees and support/load-share difference <=0.10. These targets
are not equivalent to achieving the 45 mm training trajectory. Contact-force
magnitude shares are load proxies, not a complete dynamic-balance criterion.

The native checker uses independent CPU physics, preserves target filtering,
slew and soft limits, and verifies the 510-feature initial observation. Its
single-start cases are a distinct check from the multi-start GPU matrices.

## Stop, warm-start and resume

```powershell
& $env:WEEK7_PYTHON -B code/status.py runs/<run>
New-Item -ItemType File runs/<run>/STOP
```

The STOP file requests a checkpoint and exit after the current update.
`--init-from <checkpoint>` loads weights/normalizers with a fresh optimizer
and local curriculum; `--resume <checkpoint>` preserves optimizer/curriculum
within the same reward round. `--iterations` means additional updates.
Checkpoints are written every 25 updates and at completion.

## Portable actor export and hardware context

```powershell
& $env:WEEK7_PYTHON -B code/export_actor.py policies/<candidate>/model_<iteration>.pt --output-dir policies/<candidate>
```

Exported `actor.npz` contains the deterministic network and normalizer;
`actor_contract.json` records its checksum, 510-input schema and arm ownership.
`actor_parity.json` compares NumPy output with checkpoint tensors. The exported
`NumpyActor` helper itself requires only NumPy, not CUDA or Torch.

This is an inference artifact, not a motor-control driver. Moon's 460-input
R25 runner cannot load it unchanged. The user now uses LSM6DSV16X; its actual
connection and mounting transform are unverified here. Read the
[compatibility notes](../docs/Hardware_Compatibility.md).

Keep local runs, caches, virtual environments and transient logs out of Git.
Publish selected checkpoints with source hashes, interface metadata and
independent results rather than every training snapshot.
