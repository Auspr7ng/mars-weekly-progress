# Round 3 - Bilateral gait consistency

The parent is Round 2 `model_1199.pt`. Its Home-arm fast-command short test
had mean whole-sole swing peaks of 32.7/25.3 mm. Passing the old walking gate
did not establish equal foot lift or balanced support/load transfer.

## Bounded experiment

- Name: `R26 Round 3 - Bilateral Gait Symmetry`.
- 256 environments, 1,200 PPO updates, seed 744.
- Warm-start actor/critic and normalizers from the parent; fresh optimizer,
  round-local counter and curriculum. No parent checkpoint is overwritten.
- Keep the absolute whole-sole trajectory target at 45 mm and the maximum
  forward command at 0.20 m/s; retain stop, reverse and both turn directions.
- Retain held Home, lift, spread, combined and asymmetric arm poses. Include
  both reflected versions of asymmetric arm references in real rollouts.
- Randomize the initial gait phase in training to reduce fixed-phase bias.
- Ramp the phase-matched reward multiplier from 0.25 to 1.0 over 800 updates;
  the absolute lift/speed objectives stay fixed.

## Phase-matched height cost

Compare a swinging foot's lowest whole-sole height with the opposite foot's
most recent sample at the corresponding swing-phase bin. Eight bins represent
local swing phase; only the middle 15-85% of swing is sampled. The reward
never asks both feet to have equal height at the same instant.

The comparison is enabled only after initial settling, with symmetric held
arm references and a nearly straight moving command. It clears its memory
on reset, an ineligible state or a command change greater than 0.005. Samples
older than four seconds expire. The squared difference is normalized by
6 mm and capped at 9; reward weight is -1.0 before the curriculum multiplier.
The existing absolute-height and worst-foot shortfall objectives remain,
so reducing the better foot's lift cannot satisfy the absolute target alone.
Toe-in cost weight increases from -1.5 to -2.0.

## New-schema soft mirror loss

The 460/98 observation prefix reuses the archived model-specific mapping,
including nominal joint offsets, action scales, turn-command signs and a
half-cycle phase shift. The appended 50/10 arm-reference features are now
reflected in their R5/L5 order with the proper joint-axis signs.

The mirror MSE coefficient is 0.20. It regularizes raw policy outputs; the
ten arm outputs remain superseded by held references and twelve leg outputs
actuate. Asymmetric arms are reflected to their counterpart, not forced into
a symmetric pose. Geometry is only approximately mirror symmetric, so this
is a soft prior. Reflected samples are **not** added to PPO likelihood data.
No mass, motor limit, contact model, joint limit or failure check is changed.

## Evaluation additions

The original walking gate stays unchanged for historical comparisons. New
per-case fields separately report:

- Mean per-trial difference in completed L/R swing peaks.
- L/R support time and support duty fractions after the 1.4 s settling period.
- Integrated contact-force magnitudes and their L/R shares.
- Signed mean body roll, RMS roll and maximum absolute roll.

Force magnitude is measured through the existing smoothed sensor. Its share
is a load-transfer proxy, not a center-of-pressure measurement or a complete
dynamic-balance criterion.

For symmetric held-arm straight-walking cases only, a separate diagnostic
requires the old walking gate plus both mean swing peaks >=30 mm, peak gap
<=5 mm, mean inward toe angles <=5 degrees each, roll RMS <=3 degrees,
support-duty difference <=0.10 and force-magnitude share difference <=0.10.
These are explicit diagnostic objectives, not hardware acceptance. A 45 mm
trajectory objective remains separate from the >=30 mm diagnostic floor.

Matched evaluations default to the same fixed initial phase for every round.
Use `--random-phase` on both parent and candidate for an additional matched
random-phase test. Compare the same seed, commands and duration, then repeat
with independent seeds and longer windows. Asymmetric arms have no forced
equal-load/height diagnostic. Stop/reverse/turn and native-physics checks are
still necessary before promotion.

## Monitoring

Use `open_training_monitor.bat` and press **Active run**. Losses now include
mirror consistency. Persisted status also records phase-pair availability
and the unweighted phase-height cost; neither replaces independent gait
measurements. Closing the monitor does not stop training.

## Startup evidence and parent baseline

[Preflight evidence](round3_preflight.json): 13 CPU tests and 20 GPU environment
checks passed. A 32-environment, eight-update smoke run completed 8,192
transitions with finite PPO losses, nonzero mirror loss and actual phase-pair
comparisons. Formal-run initial actor/critic tensors exactly match the parent.

The [parent balance matrix](r26_round2_parent_balance_matrix.json) uses the
same 84-trial six-second setup. At Home and 0.20 m/s, support duty fractions
were 0.591/0.674, contact-force-magnitude shares were 0.459/0.541 and body-roll
RMS was 3.08 degrees. These are finite-window measurements; they do not by
themselves establish the cause of the lower right-foot lift. The parent passed
0/72 symmetric-pose trials under the new combined diagnostic, while retaining
84/84 under the historical walking gate. The new diagnostic is deliberately
reported separately and does not retroactively change the old result.
