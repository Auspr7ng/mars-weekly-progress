# Round 4 checkpoint 600 - Measured experimental candidate

November 24, 2026. Retain this intermediate checkpoint for its higher bilateral
foot lift and forward-test reliability. It is **not an accepted release** and
does not replace the accepted R25 simulation baseline. The final overnight
selection is recorded separately in `config/selected_policy.json`.

The checkpoint follows 601 Round 4 updates from the preserved Round 2 parent.
Round 4 subsequently completed all 800 planned updates. The 45 mm training
reference and 0.20 m/s command cap are targets, not achieved guarantees.

## Independent checks

| Check | Walking passes | Survival | Symmetric diagnostic passes |
| --- | --- | --- | --- |
| 6 s, seed 1742, four starts per pose/speed | 84/84 | 84/84 | 0/72 |
| 12 s, seed 2742, eight starts per pose/speed | 168/168 | 168/168 | 0/144 |
| 12 s, seed 3742, randomized starting phase | 84/84 | 84/84 | 0/72 |
| 20 s, seed 6742, four starts per pose/speed | 84/84 | 84/84 | 0/72 |
| Independent CPU MuJoCo, 8 s, three poses/speeds | 9/9 | 9/9 | Not this check's criterion |

All GPU matrices contain seven held arm poses and commands 0.06, 0.12 and
0.20 m/s. No termination was recorded. The symmetric diagnostic excludes the
deliberately asymmetric arm pose. It remains entirely unmet despite the old
walking gate passing. Native initial observation parity error was 7.45e-9.

The suite manifest is [r26_round4_checkpoint600_suite.json](r26_round4_checkpoint600_suite.json).
Its member JSON files contain every pose/speed result and the exact criteria.
These are finite, matched simulation tests. Seed sets reused for checkpoint
selection are not a blind generalization benchmark or evidence of hardware balance.

After the experimental choice was recorded, a reserved seed-8742 randomized-phase
twenty-second matrix passed 84/84 walking trials, with no termination and 0/72
strict diagnostic passes. See the [post-selection audit](r26_round4_checkpoint600_postselection_random20.json).

## Home arms at the 0.20 m/s command

The following comparison uses the same 12-second, seed-2742 test.

| Measured quantity | Round 2 checkpoint 1199 | Round 4 checkpoint 600 |
| --- | --- | --- |
| Mean actual forward speed, m/s | 0.1802 | 0.1814 |
| Mean completed whole-sole swing peaks L/R, mm | 32.69 / 26.29 | 36.99 / 32.58 |
| Mean per-trial peak gap, mm | 6.40 | 4.42 |
| Mean inward toe angles L/R, degrees | 6.18 / 6.79 | 4.01 / 6.57 |
| Body roll RMS, degrees | 3.06 | 3.24 |
| Maximum lateral drift, m | 0.0685 | 0.0689 |

The lower right-foot mean rises by 6.29 mm in this case, but inward toe yaw
and roll still miss the strict 5-degree/3-degree limits. Mean peaks do not
mean every completed step reaches that height. In the 20-second Home fast
case the mean peaks are 36.93/33.30 mm, speed 0.1811 m/s and drift 0.104 m.
The 45 mm objective remains unachieved.

Sources: [matched parent](r26_round2_parent_balance_extended_matrix.json),
[candidate 12 s](r26_round4_checkpoint600_extended.json) and
[candidate 20 s](r26_round4_checkpoint600_holdout20.json).

## Stop, reverse and turning

All 28 thirty-second held-pose sequences survived. Every stop, forward and
left/right turn segment passed. **Reverse passed 0/28**, so complete-sequence
passes were also **0/28**. At command -0.04 m/s, mean measured reverse speeds
across poses ranged from -0.0257 to -0.0311 m/s, below the unchanged -0.032
m/s magnitude threshold. The parent passed 20/28 complete sequences.

This is a real reverse-tracking regression; surviving or turning correctly
does not erase it. See [transition results](r26_round4_checkpoint600_transitions.json).
Metric v2 uses net planar displacement for pure turns, allowing intended sway.
The earlier parent v1 report remains preserved; parent/candidate comparisons
use v2. Forward and bilateral gates were not relaxed.

## Evidence and use

The [selected preview](videos/r26_round4_checkpoint600_selected_preview.mp4) contains three actual
six-second rollouts, with full-body/leg views and English labels. The JSONL
trace records each scene's speed, foot clearance and termination status. It
is a visual example, not a substitute for the longer matrices.
The earlier [suite preview](videos/r26_round4_checkpoint600_preview.mp4) is
preserved; the selected recording uses wider leg framing with the same actor,
physical model and commands.

Held references remain fixed within each tested episode. Continuous arm
motion while walking, perturbation recovery and hardware walking are untested.
The physical model, actuator limits, action scales and failure checks are
unchanged. A separate native diagnostic of the parent found a lower right
sole center as well as lower whole-sole clearance; toe tilt alone does not
explain that parent's height difference. The restrictive local IK audit is
not a proof of dynamic infeasibility.

Moon's runner expects 460 actor inputs and its pinned driver uses BNO055.
This actor expects 510 inputs, retains 22 raw outputs/history values, applies
twelve leg outputs and obtains ten arm targets from held references. The
current IMU is **LSM6DSV16X**. Driver/mount conversion, body-speed estimation,
physical Home alignment and hardware policy-interface integration remain
unvalidated; read the [compatibility reference](../docs/Hardware_Compatibility.md).
