# Round 2 - Toe heading and higher bilateral foot lift

This iteration follows the visible toe-in gait and unequal foot clearance in
the first R26 candidate. Targets below are objectives, not achieved results.

## Training setup

- Name: `R26 Round 2 - Toe Heading and 45 mm Foot Lift`.
- Parent: `r26_clearance_speed_arms_20261119_01/model_799.pt`.
- Fresh optimizer and round-local iteration/curriculum counters; actor and
  critic weights and observation normalizers are loaded from the parent.
- 256 parallel environments, 1,200 PPO updates, seed 743; curriculum ramps
  over 800 updates, leaving 400 updates at the final target.
- Whole-sole swing trajectory target grows from 35 to 45 mm. The reward uses
  the worse-tracking foot and the largest active mid-swing deficit, so a good
  stance foot cannot hide an insufficiently lifted swing foot.
- Forward command cap grows from 0.18 to 0.20 m/s. Speed is not increased
  further until bilateral stepping and collision behavior improve.
- All seven held arm pose classes remain at their full available amplitude
  range, including combined and asymmetric poses; 25% of resets are Home.

## Toe-in correction

Signed toe yaw is measured from each sole's local -Y axis relative to the
body heading and the commanded turning foot path. The Home orientation check
confirms this axis is nearly forward. In L/R order, negative left-foot error
and positive right-foot error are inward rotations.

A continuous heading-error cost supplements the original heading reward.
An additional inward-angle cost begins beyond 0.035 rad (about 2 degrees).
Foot separation and hand-clearance penalties increase. These are soft reward
terms: hip yaw is not locked and turning remains in the command distribution.
All contact/failure checks, dynamics and physical limits are retained.

The old 460/98 mirror loss remains disabled. A correctly conditioned mirror
objective has not been implemented; symmetric per-foot measurements and
bilateral reward shaping do not imply mirror-loss validation.

## Checks and next evaluation

Four contract tests cover the schema, transfer, round targets and toe-in
signs. Eleven GPU environment checks cover target ownership, finite rewards,
the toe axis and physics. A two-update, 16-environment warm-start smoke run
completed 1,024 transitions with finite PPO losses.

Repeat the seven-pose, three-speed matrix on the parent and new candidate
with the same seed. The evaluator now additionally reports per-foot mean
absolute heading error and inward toe angle in degrees. Missing post-settle
samples are explicitly counted; zero-valued speed/error aggregates from
early failures must not be interpreted as good tracking. Compare both feet's
completed swing peaks, speed, step counts, drift, slip and self-contact.
The existing exploratory walking gate remains unchanged for comparability;
it is not proof of attaining the new 45 mm target.

Use longer and multi-seed tests, stop/reverse/turn tests and native MuJoCo
validation before promotion. No hardware deployment is authorized by a
reward increase.

## Live monitoring

Double-click `open_training_monitor.bat` in Week7. The English desktop window
shows run name, state, progress, elapsed time, estimated remaining time,
curriculum targets, losses, reward/episode-length charts and latest checkpoint.
It reads local files every two seconds without loading CUDA. Use `Active run`
to select the manifest's current run, or select an earlier run from the list.
The display flags stale running reports and checks Windows process presence;
PID reuse remains possible, so heartbeat age is also shown.

The [preflight report](round2_preflight.json) records startup evidence. The
[parent heading matrix](r26_candidate799_heading_matrix.json) reproduced the
previous survival/walking counts, with mean-speed differences below 0.0001 m/s,
and adds toe-yaw measurements. At Home and 0.06 m/s, mean inward angles were
5.53 degrees left and 6.92 degrees right; at 0.20 m/s they were 15.57 and
10.12 degrees. These are short-test baseline measurements, not Round 2 results.
