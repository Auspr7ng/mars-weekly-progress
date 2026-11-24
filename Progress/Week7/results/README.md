# Week7 results and evidence

Start with the [overnight iteration decision](Overnight_Iteration_Review.md)
and [selected checkpoint review](Round4_Checkpoint600_Review.md). The default
experimental bundle is [Round 4 checkpoint 600](../policies/r26_round4_checkpoint600/README.md);
[the selection manifest](../config/selected_policy.json) identifies its hashes.
No R26 model is accepted as a replacement for the released R25 baseline.

## Final comparison

| Evidence | Purpose |
| --- | --- |
| [Completed training records](overnight_training_runs.json) | Actual completion, deliberate stop, failure/recovery and finite recorded losses |
| [Selected model suite](r26_round4_checkpoint600_suite.json) | Short/long/random-phase/twenty-second matrices, native checks and transitions |
| [Reserved post-selection audit](r26_round4_checkpoint600_postselection_random20.json) | New seed 8742, twenty seconds and randomized phase, after the choice was recorded |
| [Round 4 final suite](r26_round4_final799_suite.json) | Later checkpoint regression retained as evidence |
| [Round 6 final suite](r26_round6_final399_suite.json) | Heading/bilateral progress and long-test drift regression |
| [Round 5 stop review](Round5_Early_Stop_Review.md) | Why 409 of 600 planned updates were sufficient to stop this branch |
| [Selected preview](videos/r26_round4_checkpoint600_selected_preview.mp4) | Actual eighteen-second video and adjacent per-step trace |

Each suite JSON links all member reports. The selected model's original
preview is preserved; its selected preview uses wider leg framing. The actor,
physics and commands are unchanged. Training checkpoints and transient runtime
files remain under ignored local `runs/`; the selected model is published.

## Startup and recovery checks

- Initial [preflight](preflight.json) and [GPU checks](environment_check.json)
  document the first 510/108/22 interface and exact weight migration.
- [Round 6 preflight](round6_preflight.json) records 18 CPU invariants,
  20 GPU contract checks and eight finite PPO startup updates.
- Three additional Windows I/O regression checks pass, including an actual
  reader lock that blocks replacement until the handle closes. Combined
  current CPU coverage is 21 passing checks.
- [Interruption record](round6_interruption.json) and
  [resumed startup checks](round6_resume_preflight.json) preserve the original
  Windows publication failure, intact checkpoint and optimizer/curriculum resume.

These are software checks, not gait acceptance. Source snapshots, dependency
versions and source hashes are retained in the selected bundle and run summary.

## Historical references

- [Round 1 review](Candidate799_Review.md), [Round 2 review](Round2_Candidate1199_Review.md)
  and [Round 3 review](Round3_Candidate1199_Review.md).
- [Round 4 plan](Round4_Plan.md), [Round 5 plan](Round5_Plan.md) and
  [Round 6 plan and recovery](Round6_Plan.md).
- [Matched parent extended matrix](r26_round2_parent_balance_extended_matrix.json),
  [parent random-phase matrix](r26_round2_parent_random_phase_matrix.json),
  [parent twenty-second matrix](r26_round2_holdout20_matrix.json),
  [native parent check](r26_round2_native_check.json) and
  [parent transition v2](r26_round2_transition_check.json).

The transition v1 parent result is preserved separately. V2 uses net planar
motion during turns so ordinary sway is not counted as translational tracking
failure; forward/reverse/stop and historical gait gates remain unchanged.
Contact-load shares are proxies, not center-of-pressure or full balance proof.
The fixed-root local inverse-kinematics audits are restrictive local searches,
not dynamic infeasibility proofs. Their hypothetical larger action gains were
not used in any training round.
