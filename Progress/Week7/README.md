# Week 7 - Higher foot lift, faster walking and held arm poses

Week7 introduces experimental R26 locomotion training after the Week6 Xbox
arm/hand controller. Code, documentation and video labels are English.
R25-300 remains the accepted flat-ground simulation baseline.

## Current work

Round 4 is the current bounded experiment. It addresses Round 3's reduction
in foot lift and long-test lateral drift, while retaining different held arm
poses and the 0.20 m/s command target. Its target is not an achieved result.
See the [Round 4 plan](results/Round4_Plan.md), [Round 3 decision](results/Round3_Candidate1199_Review.md)
and [active-run manifest](config/active_run.json). The manifest identifies the
run; `runs/<name>/status.json` supplies its latest local state.

The user confirmed a new **ST LSM6DSV16X IMU** on November 24. Moon's pinned
R25 runner still uses a historical BNO055 interface and 460-input actor.
Week7 has 510 actor inputs and is not a drop-in replacement. Read the
[hardware compatibility reference](docs/Hardware_Compatibility.md) before
using these results in the lab. No Week7 hardware run has been validated.

## Open the live monitor

Double-click [open_training_monitor.bat](open_training_monitor.bat). The
English window shows the run name, progress, ETA, reward/episode curves,
losses, curriculum and latest checkpoint. Select **Active run** to switch to
the current manifest. Closing the monitor does not stop training.

## Training history and evidence

| Round | Main change | Outcome |
| --- | --- | --- |
| 1: 800 updates | Add held-arm references; ramp speed and clearance | 19/84 short walking passes; [review](results/Candidate799_Review.md) |
| 2: 1,200 updates | Toe heading and stronger bilateral clearance | 84/84 short and 168/168 longer walking passes; unequal lift remains; [review](results/Round2_Candidate1199_Review.md) |
| 3: 1,200 updates | Phase-paired height cost and soft mirror prior | Toe-in improves, foot lift regresses; 160/168 longer walking passes; not promoted; [review](results/Round3_Candidate1199_Review.md) |
| 4: 800 planned updates | Raise-only pairing, leg-only mirror, target-aligned lift gating and lateral bias | In progress; independent measurements determine candidate selection |

These walking gates require bilateral stepping and progress, not just
survival. The stricter bilateral diagnostic is reported separately; neither
the old walking gate nor training reward proves 45 mm lift or hardware balance.

## Programs and models

- [Code and command guide](code/README.md): train, evaluate, record video,
  native-physics checks and numerical actor export.
- [Preserved Round 2 reference](policies/r26_round2/README.md): checkpoint,
  historical run metadata, NumPy actor and export-parity report.
- [Progress record](progress/README.md) and [result index](results/README.md).
- `runs/`: ignored local checkpoints, training source/configuration snapshots,
  TensorBoard data, per-update metrics and status.

The policy contract is 510 actor inputs, 108 critic inputs and 22 raw outputs.
Ten held-arm reference offsets are appended per history frame; twelve leg
outputs actuate and held references supersede the ten arm outputs. References
are fixed within an episode. Continuous arm changes while walking are not
validated by the held-pose matrix.

The frozen physical model, masses, actuator limits and failure checks remain
unchanged. Before promotion, independently check longer/multiple-seed tests,
stop/restart, reverse, both turns and native MuJoCo. Body-velocity estimation,
new-IMU mounting/driver conversion and physical Home alignment are separate
hardware requirements.
