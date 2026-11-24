# Round 2 final-candidate review

The second run completed 1,200 PPO updates and 9,830,400 transitions in
5,679.7 seconds (94.7 minutes). All 1,200 logged PPO loss records are finite.
The final model is `model_1199.pt`. It remains an experimental simulation candidate.

## Matched six-second matrix

The parent and candidate use seven held arm poses, three forward commands
(0.06, 0.12 and 0.20 m/s), four starts per case and seed 1742: 84 trials each.

| Metric | Parent R26-799 | Round 2 R26-1199 |
| --- | --- | --- |
| Survival | 80/84 | 84/84 |
| Exploratory walking gate | 19/84 | 84/84 |
| Self-contact terminations | 4 | 0 |

The unchanged walking gate requires survival, mean forward speed at least
80% of the command after 1.4 seconds, at least two completed swings per foot,
mean completed swing peaks at least 20 mm on **both** feet, and lateral drift
below 0.15 m. It does **not** test attainment of the new 45 mm training target.

## Longer independent-seed forward check

A second matrix uses seed 2742, eight starts per case and twelve seconds per
trial. All 168 trials survived and passed the same exploratory walking gate;
there were no tilt, height, ground, self-contact or nonfinite terminations.
At Home and the 0.20 m/s command, mean measured speed was 0.180 m/s,
mean completed swing peaks were 32.7/26.3 mm and inward toe yaw was 6.2/6.8
degrees. This strengthens the forward held-pose evidence while leaving the
uncovered stop/turn/reverse, native-physics and hardware cases unvalidated.

[Longer independent-seed matrix](r26_round2_candidate1199_extended_matrix.json).

## At the 0.20 m/s command

| Arm pose | Mean measured speed (m/s) | Mean completed whole-sole swing peak L/R (mm) | Mean inward toe yaw L/R (degrees) | Walking passes |
| --- | --- | --- | --- | --- |
| home | 0.181 | 32.7 / 25.3 | 6.3 / 6.7 | 4/4 |
| lift | 0.181 | 29.9 / 29.6 | 7.0 / 7.9 | 4/4 |
| spread | 0.179 | 31.4 / 25.5 | 6.1 / 6.8 | 4/4 |
| lift_spread | 0.178 | 28.8 / 29.2 | 6.6 / 7.8 | 4/4 |
| full_lift | 0.182 | 29.6 / 30.0 | 8.0 / 8.2 | 4/4 |
| full_spread | 0.178 | 30.2 / 26.6 | 5.8 / 6.8 | 4/4 |
| asymmetric | 0.179 | 31.8 / 27.8 | 7.7 / 7.7 | 4/4 |

For Home arms at the fast command, inward toe yaw decreased from 15.6/10.1
to 6.3/6.7 degrees, mean swing peaks increased from 29.9/19.8 to 32.7/25.3 mm,
and measured speed increased from 0.173 to 0.181 m/s. Raised-and-spread arms
now passed all four fast trials, compared with four self-contact failures
in the parent.

## Remaining limitations

- Toe-in is reduced, not eliminated. At fast commands, mean inward angles
  remain roughly 6-8 degrees across poses.
- Bilateral foot lift improved, especially on the right, but 45 mm is still a
  trajectory objective rather than achieved mean completed-swing clearance.
  Home retains a left/right difference; other raised-arm poses are more balanced.
- The model does not yet sustain a measured 0.20 m/s across these cases.
- Passing this short matrix is not native-MuJoCo or hardware acceptance.
  Stop/restart, reverse, turn, perturbation and continuous arm-transition tests
  are not covered here.

## Evidence

- [Parent matrix with toe-heading measurements](r26_candidate799_heading_matrix.json)
- [Round 2 matched matrix](r26_round2_candidate1199_matrix.json)
- [Selected single-trial preview](videos/r26_round2_candidate1199_preview.mp4)

The selected preview uses separate single-environment GPU rollouts and native
MuJoCo rendering of their states, with full-body and elevated foot views.
It is qualitative evidence, not the matrix itself or a native-physics test.
No additional training run is started by this review.

Final checkpoint SHA-256: `b3bb3f299617fc48959a78cd354f47fd40d3c835b0ba1ee9a1cbbd82cd697ddb`.
