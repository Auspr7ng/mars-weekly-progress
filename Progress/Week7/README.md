# Week 7 - Higher foot lift, faster walking and held arm poses

Week7 introduces experimental R26 locomotion training after the Week6 Xbox
arm/hand controller. Code, documentation and video labels are English.
R25-300 remains the accepted flat-ground simulation baseline.

## Selected experimental progress

The November 24 overnight comparison selects **Round 4 checkpoint 600** for
held-pose forward reliability and higher bilateral lift. It passes 84/84
six-second, 168/168 twelve-second, 84/84 random-phase and 84/84 twenty-second
matched walking tests, plus 9/9 independent native cases. At Home and a
0.20 m/s command, the twelve-second measurement is 0.1814 m/s actual speed
and 36.99/32.58 mm mean completed swing peaks, compared with 32.69/26.29 mm
for the Round 2 parent. These are measured means, not every-step guarantees.

The final checkpoint is not automatically the best: Round 4 checkpoint 799
regresses to 72/84 twenty-second passes; Round 6 checkpoint 399 regresses to
160/168 twelve-second passes. Round 5 is deliberately stopped after 409 of
600 planned updates because its foot lift regresses. Round 6 finishes a
400-update retained path after a documented Windows status-file recovery.

Read the [overnight decision](results/Overnight_Iteration_Review.md),
[candidate review](results/Round4_Checkpoint600_Review.md),
[model bundle](policies/r26_round4_checkpoint600/README.md) and
[selection manifest](config/selected_policy.json). The bundle includes PT/NumPy
weights, export parity, an immutable training-source archive and checksums.
The [selected preview](results/videos/r26_round4_checkpoint600_selected_preview.mp4)
shows three actual rollouts with English labels.

This remains an experimental simulation candidate. The strict bilateral
diagnostic is unmet; inward toe yaw, unequal lift and reverse underspeed remain.
All 28 transition sequences survive and their stop/forward/turn segments pass,
but reverse fails 28/28, so no full sequence passes. It does not replace the
accepted R25 baseline or constitute a hardware release. The 45 mm reference
remains an unachieved objective.

The user confirmed a new **ST LSM6DSV16X IMU** on November 24. Moon's pinned
R25 runner still uses a historical BNO055 interface and 460-input actor.
Week7 has 510 actor inputs and is not a drop-in replacement. Read the
[hardware compatibility reference](docs/Hardware_Compatibility.md) before
using these results in the lab. No Week7 hardware run has been validated.

## Open the live monitor

Double-click [open_training_monitor.bat](open_training_monitor.bat). The
English window shows the run name, progress, ETA, reward/episode curves,
losses, curriculum and latest checkpoint. Select **Active run** to switch to
the current manifest. Closing the monitor does not stop training. The final
active run is the completed Round 6 resume segment; its 74 updates continue
from absolute iteration 326 to 399. The historical
[run summary](results/overnight_training_runs.json) remains available without
the ignored local run folders.

## Training history and evidence

| Round | Main change | Outcome |
| --- | --- | --- |
| 1: 800 updates | Add held-arm references; ramp speed and clearance | 19/84 short walking passes; [review](results/Candidate799_Review.md) |
| 2: 1,200 updates | Toe heading and stronger bilateral clearance | 84/84 short and 168/168 longer walking passes; unequal lift remains; [review](results/Round2_Candidate1199_Review.md) |
| 3: 1,200 updates | Phase-paired height cost and soft mirror prior | Toe-in improves, foot lift regresses; 160/168 longer walking passes; not promoted; [review](results/Round3_Candidate1199_Review.md) |
| 4: 800 updates | Raise-only pairing, leg-only mirror, target-aligned lift gating and lateral bias | Select checkpoint 600; final 799 drifts in longer tests |
| 5: 409/600 planned updates | 35 mm reference and stronger bilateral/heading costs | Deliberately stopped; toe-in improves but foot lift regresses |
| 6: 400 retained updates | More vertical gait freedom and stronger leg-only mirror prior | Completed after I/O recovery; some bilateral diagnostics improve, forward reliability regresses |

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
unchanged. Longer/multiple-seed tests, stop/restart, reverse, both turns and
native MuJoCo are documented; remaining failures prevent release acceptance.
Body-velocity estimation,
new-IMU mounting/driver conversion and physical Home alignment are separate
hardware requirements.
