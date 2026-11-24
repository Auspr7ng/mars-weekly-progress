# Overnight iteration decision - November 24, 2026

Select **Round 4 checkpoint 600** as the default experimental progress model.
It raises bilateral foot lift while retaining the strongest measured held-pose
forward reliability among the tested candidates. Later checkpoints improve
some individual measurements but regress on long-test drift. No R26 model is
accepted as a replacement for the released R25 baseline or as a hardware release.

Start with the [selection manifest](../config/selected_policy.json),
[model bundle](../policies/r26_round4_checkpoint600/README.md),
[detailed candidate review](Round4_Checkpoint600_Review.md) and
[selected video](videos/r26_round4_checkpoint600_selected_preview.mp4).
All code, guides and video labels are English.

## What actually ran

| Round | Training outcome | Decision |
| --- | --- | --- |
| 4 | 800 updates completed | Retain intermediate checkpoint 600, after 601 updates |
| 5 | Deliberately stopped after 409 of 600 planned updates | Repeated height regression outweighs improved toe heading |
| 6 | Retained path completes 400 updates after an I/O failure/recovery | Keep results as evidence; do not select its drifting final model |

The original Round 6 process attempts 331 updates and fails during atomic
Windows status-file publication. Checkpoint 325 retains 326 updates. A new
directory resumes 74 updates and saves checkpoint 399, with environment step
counter 12,800. Five unpersisted updates are recomputed, so this round executes
405 updates in total while its retained path contains 400. Optimizer,
normalizers and curriculum counter are restored; simulator state/RNG are not
a bitwise continuation. No reward or physical-model change is made for recovery.

The fix adds bounded replacement retries for transient reader locks. An actual
Windows open reader blocks replacement in the regression test, then the write
succeeds after the reader closes. Persistent denial still raises and preserves
the original destination. Eighteen training invariants and three I/O regression
checks pass. Original Round 6 GPU contract checks pass 20/20; resumed startup
losses are finite and the curriculum remains at its final level.

See [training records](overnight_training_runs.json),
[failure evidence](round6_interruption.json),
[resume checks](round6_resume_preflight.json) and
[Round 5 stop review](Round5_Early_Stop_Review.md). All recorded training losses
in these runs are finite. Source snapshots and local checkpoints remain intact.

## Matched gait comparison

Each forward matrix contains seven held arm poses and commands 0.06, 0.12
and 0.20 m/s. Short tests use seed 1742, twelve-second tests seed 2742,
random-phase twelve-second tests seed 3742 and twenty-second tests seed 6742.
The original walking gates and stricter bilateral diagnostic are unchanged.

| Model | 6 s walking | 12 s walking | Random-phase 12 s walking | 20 s walking | Native 8 s | Complete transition sequences |
| --- | --- | --- | --- | --- | --- | --- |
| Round 2 parent 1199 | 84/84 | 168/168 | 82/84 | 80/84 | 9/9 | 20/28 |
| **Round 4 checkpoint 600** | **84/84** | **168/168** | **84/84** | **84/84** | **9/9** | **0/28** |
| Round 4 final 799 | 84/84 | 168/168 | 84/84 | 72/84 | 9/9 | 4/28 |
| Round 5 stopped 408 | 84/84 | Not run | Not run | Not run | Not run | Not run |
| Round 6 final 399 | 84/84 | 160/168 | 81/84 | 60/84 | 9/9 | 4/28 |

All three fully checked new candidates survive their matched forward matrices.
Their failed walking gates therefore reveal performance failures rather than
being hidden by automatic resets or survival-only scoring. The Round 2 random
phase parent has one self-contact termination; its walking passes are 82/84.

Round 4 final 799 fails twelve twenty-second trials due to drift: Home, lift
and full-lift arms at 0.20 m/s. Round 6 final 399 fails eight Home fast trials
in the twelve-second matrix, then twenty-four trials in the twenty-second
matrix. Home fast twenty-second drift reaches 0.267 m. This prevents choosing
the final Round 6 model despite improved toe heading.

The selected model's strict symmetric diagnostic is 0/72, 0/144, 0/72 and
0/72 in these four matrices. Round 6 achieves 7/144, 10/72 and 8/72 in the
long/random/twenty-second matrices, showing partial progress, while its
forward reliability regresses. Passing a few diagnostic cases is not a release.

Source suites: [selected](r26_round4_checkpoint600_suite.json),
[Round 4 final](r26_round4_final799_suite.json) and
[Round 6 final](r26_round6_final399_suite.json). The parent sources are indexed
in [the result guide](README.md). Matched seed sets used during selection are
finite comparisons, not a blind generalization benchmark.

## Height, heading and speed

Home arms, 0.20 m/s command, twelve-second matrix:

| Model | Actual m/s | Mean completed peaks L/R, mm | Mean per-trial gap, mm | Inward toe L/R, degrees | Roll RMS, degrees | Max drift, m |
| --- | --- | --- | --- | --- | --- | --- |
| Round 2 parent | 0.1802 | 32.69 / 26.29 | 6.40 | 6.18 / 6.79 | 3.06 | 0.0685 |
| **Round 4 checkpoint 600** | **0.1814** | **36.99 / 32.58** | **4.42** | **4.01 / 6.57** | **3.24** | **0.0689** |
| Round 4 final 799 | 0.1804 | 38.39 / 32.23 | 6.16 | 4.96 / 6.12 | 3.28 | 0.121 |
| Round 6 final 399 | 0.1806 | 38.01 / 32.74 | 5.26 | 3.11 / 5.28 | 3.22 | 0.167 |

Selected-model fast twenty-second results by held pose:

| Pose | Actual m/s | Mean completed peaks L/R, mm | Max drift, m |
| --- | --- | --- | --- |
| Home | 0.1811 | 36.9 / 33.3 | 0.104 |
| Lift | 0.1804 | 33.3 / 36.7 | 0.063 |
| Spread | 0.1820 | 37.7 / 34.9 | 0.036 |
| Lift and spread | 0.1790 | 34.1 / 36.9 | 0.030 |
| Full lift | 0.1805 | 33.9 / 37.3 | 0.140 |
| Full spread | 0.1823 | 38.2 / 36.1 | 0.068 |
| Asymmetric | 0.1796 | 34.8 / 36.4 | 0.056 |

These are whole-sole completed-swing means, including startup, not guarantees
that every step reaches the listed height. The 45 mm reference remains unmet.
The 0.20 m/s command cap exceeds the accepted R25 cap of 0.12 m/s; it is not
a certified maximum speed. The later rounds chiefly alter gait quality, not
the already established approximately 0.18 m/s Home fast speed.

The parent's native diagnostic finds a lower right sole center as well as
lower whole-sole clearance; toe tilt alone does not explain that case. Unequal
height does not itself prove loss of balance. Roll, contact-load proxies,
support duty, drift and failures are measured separately. Contact-force shares
are not center-of-pressure or a full balance proof. Fixed-root local IK results
are not a proof that dynamic higher steps are impossible.

## Reserved audit and action transitions

The selection is recorded at 02:46:24 Indianapolis time, before an additional
seed-8742 randomized-phase twenty-second audit. That audit passes **84/84**
walking trials, with no termination and 0/72 strict symmetric passes. The
selected model therefore passes all 504 recorded forward trials across five
matrices; this is a count of sampled tests, not a formal reliability bound.
See [the post-selection audit](r26_round4_checkpoint600_postselection_random20.json).

All selected-model thirty-second action sequences survive. Every stop,
forward and both turn segments pass 28/28. Reverse passes **0/28**: at a
-0.04 m/s command, measured means range from -0.0257 to -0.0311 m/s across
poses, below the unchanged 80% threshold. Complete-sequence passes are thus
0/28. This reverse regression is an explicit limitation, not a successful
whole-controller acceptance.

Transition metric v2 measures net planar displacement during pure turns,
allowing planned sway. Parent v1 evidence is retained separately. Parent and
new candidates use v2; forward/reverse/stop and bilateral gates are unchanged.

## Published artifact and laboratory context

The selected bundle includes the original PT checkpoint, deterministic NumPy
actor, interface contract, numerical parity report, historical run metadata,
18-file hash-verified training-source archive and file checksums. NumPy/Torch
maximum output difference is 3.28e-7 across 64 seeded probes. All 23 floating
actor/critic state tensors are finite. Export agreement is not driver validation.

The selected video contains three actual six-second rollouts and 900 traced
control steps with no termination. Its wider leg view makes both feet visible;
the earlier suite preview is also preserved. Check the [monitor](../open_training_monitor.bat)
for local historical runs and the completed final resume segment.

The current IMU is **ST LSM6DSV16X**. Moon's pinned repository is preserved as
the hardware reference; its BNO055 driver and 460-input actor loader do not
directly support this new sensor and 510-input policy. Week7 retains 22 raw
outputs/history values but applies twelve leg outputs, with ten arm targets
owned by held references. New IMU driver/mount conversion, body-speed estimation,
Home alignment and hardware integration are unvalidated. The fixed synthetic
0.06 m/s velocity observation is distinct from the command and its role in
the earlier jerky lab gait remains unisolated. No hardware motor command or
new IMU register write was performed. Read [the compatibility reference](../docs/Hardware_Compatibility.md).

No further training is launched after the bounded Round 6 comparison. The
next reward iteration should test accumulated straight-line drift and reverse
tracking alongside lower-foot clearance, and isolate vertical-margin/mirror
effects rather than attributing this multi-change comparison to one cause.
The selected checkpoint, unchanged evaluation gates and hardware measurements
provide the starting point; remaining failures are preserved for that work.
