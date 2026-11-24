# Round 5 - Bilateral 35 mm reference comparison

This companion experiment starts from Round 4's checkpoint 200 while the
original 45 mm experiment continues. It tests whether a lower trajectory
reference and stronger bilateral consistency produce better measured minimum
foot lift, heading and drift. The evaluation gates are unchanged.

## Setup

- Parent: Round 4 `model_200.pt` (201 completed updates, not the final model).
- 512 environments, 600 updates, seed 746, regularizer ramp over 300 updates.
- 35 mm whole-sole trajectory reference; 0.20 m/s forward command cap.
- Raise-only corresponding-phase weight -0.8 and leg-only mirror coefficient
  0.10. Heading/toe-in weights are -4.5/-4.0.
- Keep target-aware lift gating, absolute shortfall, lateral-bias shaping,
  random initial phase and both asymmetric-arm counterparts from Round 4.
- Fresh optimizer and local counters; actor/critic/normalizers warm-start from
  the preserved intermediate parent. Mechanical limits, motor speeds/effort,
  action scales and failure checks are unchanged.

## Rationale and limits

At Round 4 checkpoint 200, all 84 short forward trials passed the historical
walking gate. Fast Home peaks were about 38.1/28.8 mm; the right foot remained
lower despite the higher absolute lift incentive.

A separate local inverse-kinematics audit with fixed root and fixed foot
orientation did not find all requested high-lift poses inside the policy's
residual bounds. A hypothetical residual-gain comparison still did not solve
all cases. These are **local search results under restrictive static
assumptions**, not proofs of dynamic infeasibility. The larger hypothetical
gains are not applied to either training experiment.

The 45 mm value remains an unachieved stretch objective. Selecting a 35 mm
training reference does not establish that either foot actually reaches it.
Candidate choice must use the same measured-height, gap, toe-angle, support,
roll, drift and survival evidence across both experiments. This is a targeted
comparison, not a one-factor causal ablation, because consistency and heading
weights also change.

The selected model will be evaluated with fixed/random initial phases,
longer forward trials, native CPU physics and stop/reverse/turn sequences.
Publish both progress and unmet conditions; do not reclassify a training
reward increase or an easier reference as passing the unchanged diagnostics.

## Recorded stop decision

This branch was deliberately stopped after 409 of the 600 planned updates.
Heading improved, while measured foot lift regressed in repeated short checks.
The stopped checkpoint and metrics were preserved locally. Read the
[early-stop review](Round5_Early_Stop_Review.md) for measurements and scope.
