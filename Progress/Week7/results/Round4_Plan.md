# Round 4 - Preserve bilateral lift while improving heading and drift

Reviewed and started November 24, 2026 after Round 3 independent evaluations.
Round 3 improved toe heading but reduced foot lift and did not improve the
combined bilateral diagnostic. It is not promoted as an overall replacement.

## Bounded training

- Parent: the preserved Round 2 `model_1199.pt`.
- 512 parallel environments, 800 PPO updates, seed 745.
- Fresh optimizer; actor/critic and normalization state warm-start from parent.
- Learning rate 5e-5, with the phase-pair multiplier ramping over 400 updates.
- Whole-sole trajectory target remains 45 mm; forward command cap remains
  0.20 m/s. Stop, reverse, turns and held arm references are retained.
- Initial gait phase is randomized, and both counterparts of asymmetric arm
  references are sampled in real physics rollouts.

## Objective revisions

1. The paired-phase cost becomes one-sided: only a foot more than 2 mm below
   the opposite foot's corresponding phase sample is penalized. The higher
   foot is not penalized for being higher. Weight decreases to -0.35.
2. Absolute shortfall follows the actual 45 mm trajectory during the central
   20-80% of swing, instead of the previous constant 75%-of-peak objective.
   Clearance and shortfall weights become +6 and -5 respectively.
3. Speed/progress/yaw tracking retain their archived equations but use a
   support/slip/lift gate based on the Week7 target rather than the archived
   24 mm trajectory. The historical 15 mm phase score remains present as a
   separate coarse stepping term; it does not replace the absolute lift goal.
4. Mirror consistency operates only on the twelve applied leg outputs,
   excluding the ten held-arm outputs. Its coefficient decreases to 0.05.
   Reflected PPO likelihood data remain disabled.
5. Heading-error and toe-in weights increase to -3.5 and -3.0. A resettable
   half-second low-pass filter penalizes sustained excess lateral velocity
   after subtracting the commanded lateral motion and planned sway. Weight
   is -0.75; normal sway and turning are not forced to zero.

These changes are a targeted objective revision, not a one-factor causal
ablation. The eventual report must compare actual gait measurements, not
cross-round training reward. Physical limits/dynamics and failure checks
are unchanged.

## Checks and candidate selection

Regression tests cover raise-only pairing and prove that arbitrary ignored
arm-output errors do not enter the new leg-only mirror loss. CPU tests,
GPU schema/physics checks and a bounded warm-start smoke run precede training.

Inspect intermediate checkpoints and the final candidate against preserved
Round 2 and Round 3 evidence. Repeat the fixed-phase and random-phase forward
matrices, and report peak-height difference, measured speed, toe yaw, support
time, load proxies, body roll, drift and failures. Do not weaken the existing
walking or bilateral diagnostic gates to claim progress. Additional stop,
reverse, turn and native-physics checks are required before promotion.

The selected checkpoint and its results will be published as experimental
evidence. No training reward establishes hardware acceptance.
