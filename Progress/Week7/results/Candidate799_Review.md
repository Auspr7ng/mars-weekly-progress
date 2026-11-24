# R26-799 preliminary review

Reviewed after the first 800-update run completed on November 19, 2026.
Training took 3,569.5 seconds (59.5 minutes) and collected 6,553,600 transitions.
The current candidate is `model_799.pt`. It is not an accepted release.

## Matched short evaluation

Both matrices use seed 1742, 4 initial states per case, 6 seconds per episode,
7 arm poses and forward commands 0.06, 0.12 and 0.20 m/s. There are 84 trials
per model. The reference is the transferred R25 initialization under the same
new held-arm interface, not the original Week4 acceptance setup.

| Metric | Transferred initialization | R26-799 |
| --- | --- | --- |
| First-episode survival | 49/84 | 80/84 |
| Exploratory walking gate | 8/84 | 19/84 |
| Self-contact terminations | 35 | 4 |

The walking gate requires survival, sustained speed at least 80% of the
command after the initial 1.4 seconds, at least two completed swings per foot,
mean completed swing peaks of at least 20 mm on each foot, and lateral drift
below 0.15 m. Survival alone is insufficient.

## At the 0.20 m/s command

| Held arm pose | Mean measured speed (m/s) | Mean completed swing peak L/R (mm) | Survival | Walking gate |
| --- | --- | --- | --- | --- |
| home | 0.173 | 29.9 / 19.8 | 100% | 0% |
| lift | 0.173 | 26.6 / 21.5 | 100% | 100% |
| spread | 0.175 | 28.1 / 18.6 | 100% | 0% |
| lift_spread | 0.172 | 24.1 / 20.5 | 0% | 0% |
| full_lift | 0.159 | 22.9 / 25.9 | 100% | 0% |
| full_spread | 0.176 | 23.3 / 19.3 | 100% | 0% |
| asymmetric | 0.167 | 32.4 / 17.7 | 100% | 0% |

The Home case improved from 0.119 to 0.173 m/s under the same command.
This is a measured speed improvement in the short new-task comparison. It
does not establish reliable 0.20 m/s speed or a hardware walking envelope.

## Remaining problems

- Clearance is asymmetric. For example, Home at the fast command averages
  29.9 mm on the left and 19.8 mm on the right. Several slow/asymmetric cases
  have only about 15-20 mm on the right. Both feet have not met the 35 mm
  trajectory target consistently.
- Raised-and-spread arms at the fast command triggered self-contact in all
  four trials. Other poses surviving the short test still often fail the
  bilateral walking gate.
- Some cases remain below the required measured speed fraction or exceed
  drift/step criteria. Training reward is not used to waive those failures.

## Suggested next iteration

Emphasize the lower-clearance foot rather than only averaging foot rewards,
restore an appropriate left/right consistency objective for the new reference
schema, and focus on combined arm poses and hand/leg collision margins.
Repeat the same matrix before considering stronger speed demands. Follow with
longer multi-seed stop, reverse, turn and native-physics checks before promotion.
No additional training run is started by this review.

## Evidence

- [Transferred initialization matrix](r25_transfer_baseline_matrix.json)
- [Candidate matrix](r26_candidate799_matrix.json)
- [Selected single-trial video](videos/r26_candidate799_preview.mp4)

The video is a separate qualitative GPU-physics rollout, with full-body and
foot-closeup views. It is not the 84-trial matrix and does not establish
native MuJoCo or hardware acceptance.
