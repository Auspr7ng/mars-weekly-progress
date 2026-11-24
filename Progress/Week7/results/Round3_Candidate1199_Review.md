# Round 3 review and iteration decision

Reviewed November 24, 2026. Round 3 completed 1,200 PPO updates and 9,830,400
transitions in 5,905.1 seconds (98.4 minutes). All 1,200 logged PPO loss records
are finite. Independent evaluations show a tradeoff, not overall improvement.
**Decision: further targeted iteration is needed; Round 3 is not promoted.**

## Matched forward results

Each cohort covers seven held arm poses and three forward commands (0.06,
0.12 and 0.20 m/s). Both models use identical seeds and test settings.

| Cohort | Parent Round 2 walking | Round 3 walking | Parent terminations | Round 3 terminations |
| --- | --- | --- | --- | --- |
| 6 s, 4 starts/case, seed 1742, fixed phase | 84/84 | 84/84 | 0 | 0 |
| 12 s, 8 starts/case, seed 2742, fixed phase | 168/168 | 160/168 | 0 | 0 |
| 12 s, 4 starts/case, seed 3742, random phase | 82/84 | 79/84 | 1 | 0 |

All fixed-phase episodes survived. The Round 3 long-cohort walking failures
were the eight fast Home cases; their group maximum lateral drift was 0.156 m,
above the 0.15 m gate. The report stores group aggregates, so this maximum
alone does not assign a per-trial failure cause. Random-phase Round 3 avoided
the parent's single self-contact termination, but passed fewer walking trials.

## Fast Home case in the 12-second fixed-phase cohort

| Measurement | Round 2 | Round 3 |
| --- | --- | --- |
| Mean measured forward speed (m/s) | 0.180 | 0.185 |
| Mean completed swing peaks L/R (mm) | 32.7 / 26.3 | 29.4 / 24.2 |
| Mean inward toe yaw L/R (degrees) | 6.2 / 6.8 | 2.0 / 5.5 |
| Support duty fractions L/R | 0.608 / 0.660 | 0.619 / 0.649 |
| Contact-force magnitude shares L/R | 0.473 / 0.527 | 0.484 / 0.516 |
| Mean per-trial swing-peak gap (mm) | 6.4 | 5.2 |
| Mean body-roll RMS (degrees) | 3.06 | 2.87 |

Both models pass 0/144 symmetric-pose trials under the separate combined
bilateral diagnostic in this cohort. This is not a claim of inevitable loss
of balance: it means the declared height, gap, toe-angle, roll, support and
load-proxy objectives are not jointly attained. Neither model achieves the
45 mm whole-sole trajectory objective consistently.

## Why the next round changes the objectives

The Round 3 phase cost squares the L/R height difference. It can shrink that
difference by raising the lower foot or lowering the higher foot. The observed
height regression is consistent with such a reward tradeoff, but this review
has not isolated its cause with a one-factor ablation. The full-output mirror
loss also includes ten outputs that do not actuate the held arms.

Inherited speed/progress reward gating still saturates around the archived
24 mm trajectory, and an inherited phase score saturates at 15 mm, while the
new clearance terms ask for 45 mm. These differing incentives warrant an
explicit lift-gate update rather than simply training the same setup longer.

Round 4 therefore starts from the preserved Round 2 checkpoint, applies a
raise-only corresponding-phase cost, restricts the softened mirror loss to
leg outputs, aligns speed/progress gating with the Week7 lift target,
strengthens absolute lift/toe-in costs and penalizes sustained lateral bias.
Physical limits and failure checks remain unchanged. See the Round 4 plan
and its independent evaluation before choosing a new experimental candidate.

## Evidence

- [r26_round2_parent_balance_matrix.json](r26_round2_parent_balance_matrix.json)
- [r26_round3_candidate1199_matrix.json](r26_round3_candidate1199_matrix.json)
- [r26_round2_parent_balance_extended_matrix.json](r26_round2_parent_balance_extended_matrix.json)
- [r26_round3_candidate1199_extended_matrix.json](r26_round3_candidate1199_extended_matrix.json)
- [r26_round2_parent_random_phase_matrix.json](r26_round2_parent_random_phase_matrix.json)
- [r26_round3_candidate1199_random_phase_matrix.json](r26_round3_candidate1199_random_phase_matrix.json)
- [Selected Round 3 GPU-rollout preview](videos/r26_round3_candidate1199_preview.mp4)

The preview is qualitative GPU-physics evidence, not a native-physics or hardware test.
