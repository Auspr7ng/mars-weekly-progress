# Reproducing the Week 4 release

## Environment and entry points

Use Python 3.11, an NVIDIA GPU/driver, and the package versions in [requirements.txt](../code/requirements.txt). The validated environment used an RTX 4070 Laptop GPU with 8 GB VRAM and PyTorch 2.7.1+cu128. The requirements file names the PyTorch release; use its compatible CUDA build rather than assuming a CPU wheel is sufficient.

`code/scripts/bootstrap.py` prepends `code/vendor` and `code/src`, so the included mjlab and Viser sources take precedence over unrelated editable checkouts. Their licenses and source notes are in [code/vendor/README.md](../code/vendor/README.md). The exact prebuilt Viser client is included because drag start/update/end behavior is part of the control contract; no Node build or browser-side package download is required.

```powershell
python code/run_demo.py --check
python code/run_demo.py
python code/run_demo.py --port 8087
```

Use another port if an existing viewer owns 8086. The launcher reuses a running viewer only when its local server record matches the checkpoint, command limits, body-follow UI version and native CPU backend. Windows launchers use `WEEK4_PYTHON` if set, otherwise Python on PATH. Linux uses the same variable through `start_demo.sh`. No launcher installs dependencies or starts training.

The browser connects to local Python. Native MuJoCo performs physical integration and the CPU actor outputs joint targets. GPU mjlab setup and the tactile implementation still require the CUDA environment. Do not describe this release as CPU-only installation.

## Artifact identity

| Artifact | SHA-256 |
| --- | --- |
| `policies/joystick_walking_r25_300/model_300.pt` | `3bf72dd99c3c75049db74b4d50c94a6f3ce270af6887a7c048a16130b1ba6202` |
| Frozen dynamics MJCF | `c1168aee2b339adc15fa8fc3338e42c729dc92f0db07891c685cf96ca604bcd2` |
| Frozen arm/reference JSON | `877c3e74609dc365e37aa49f1816be5b1ad2343b796ae00bd71303cc2a6f21f4` |

`--check` validates these identities, all training-source hashes from the selected run, CUDA availability, and the packaged drag-lifecycle API/client. The current viewer and the frozen training source have separate roles: archived source describes how the policy was trained; current viewer source includes the later joystick fixes.

The final schema is actor 460, critic 98, action 22. Commands and velocities use the task body frame, +X forward, +Y left, +Z up. Physics dt is 0.002 s, control dt is 0.02 s. Head and finger joints remain physical but are not the 22 learned outputs.

## Checks and evaluations

The controller regression uses real MuJoCo poses and actual callback logic with recorded scene handles, without operating a live web client:

```powershell
python code/scripts/check_body_joystick.py
```

The supplied evaluation scripts expose `--help` for explicit checkpoint, seed, program, duration and output selection. They do not automatically promote a checkpoint:

```powershell
python code/scripts/evaluate_commands.py --help
python code/scripts/evaluate_native_commands.py --help
python code/scripts/evaluate_joystick_control.py --help
python code/scripts/evaluate_native_joystick.py --help
python code/scripts/audit_gait_quality.py --help
python code/scripts/assess_joystick_trace.py --help
python code/scripts/assess_control_result.py --help
```

Native feedback runs its own MuJoCo integration after initialization. It does not copy subsequent GPU poses. Native actor device CPU is the web backend; GPU inference is checked separately.

The release includes the 100-episode joystick NPZ and its raw/cycle-aware reports, allowing the cycle assessment to be revisited. JSON control summaries contain all 1,200 trial outcomes. Bulk per-step control trajectories remain local, so reproducing every plot from those traces requires rerunning the corresponding evaluation. Versioned criteria are in `config/control_acceptance_phase_aware.json`; the original false v1 verdict is preserved.

## Training records and continuation

Historical run settings/status are under `results/run_metadata`. The final R25 TensorBoard event stream is under `results/training`. Open it with `View_Learning_Curves.cmd`, or:

```powershell
python -m tensorboard.main --logdir results/training --host 127.0.0.1 --port 6006
python code/scripts/plot_training_history.py
```

R25 used 512 environments, 32 control steps per rollout, seed 42, and a planned 400-update segment. It stopped after saving update 336; checkpoint 300 was evaluated and selected. The 25% nominal-start fraction complements randomized starts. Final weights inherit earlier R22 work and are not a claim of training from scratch in 300 updates.

Training is explicit and bounded. To study continuation from the accepted same-dynamics checkpoint, choose a new output directory and a finite iteration count:

```powershell
python code/scripts/train.py --stage control --revision 25 --whole-body --envs 512 --iterations 100 --resume policies/joystick_walking_r25_300/model_300.pt --run-dir logs/r25_continuation
```

This is a new experiment and requires its own evaluations. The complete original multiround checkpoint ancestry and discarded trials remain local; the release is sufficient for selected-policy playback and same-artifact continuation, not a full byte-for-byte replay of every historical training segment.

## Publication scope

The original live Week4 workspace is retained. The GitHub copy consolidates reports into English, places documents under `docs`, moves small historical run metadata into `results/run_metadata`, normalizes machine-local metadata paths, translates report-generator strings and CAD material labels, and replaces drive-specific launchers. Learned weights, physical MJCF, nominal reference and archived training source are preserved by hash. A current Viser snapshot and client are included for portability.

Original Word/Fusion files, copied external repositories, caches, intermediate checkpoint collections and bulk evaluation trajectories are omitted. [Local_Archive_Index.json](Local_Archive_Index.json) records their paths/sizes. [Publication_Record.json](Publication_Record.json) records selected original/published hashes and checks. These publication edits do not turn estimates into measured hardware parameters or distance heatmaps into pressure.

Historical CAD extraction/rebuild helpers require the original local CAD/Word inputs. Those inputs are not required for accepted-policy replay or continuation from the included dynamics. Git attributes preserve exact line endings for hashed model/reference and frozen training-source artifacts.
