# Week7 checks and evaluations

## Preflight

- Three CPU tests passed for target ranges, pose signs/asymmetry and exact
  first-layer/normalization expansion.
- [GPU environment checks](environment_check.json) passed all nine checks:
  510/108 observation dimensions, reference history, posed initial state,
  ignored policy arm outputs, twelve applied leg outputs, soft limits,
  finite full-pose physics and subset reset semantics.
- A 16-environment, two-iteration PPO smoke run completed 1,024 transitions
  with finite losses. Its transfer probe measured maximum deterministic
  differences of about 1.2e-7 for the actor and 3.8e-6 for the critic.
  This is a software startup check, not gait acceptance.

## Baseline and candidate matrices

[The initialization matrix](r25_transfer_baseline_matrix.json) uses the transferred
`model_initial.pt` before Week7 training. It is evaluated with the same held
arm interface and Week7 cadence/reference contract as future candidates. It
is a matched task-initialization comparison, not a new independent acceptance
of the original Week4 demo.

The evaluator separately reports each pose/speed case, failures, sustained
velocity, completed swing peaks, bilateral stepping, support slip and drift.
Training reward and survival do not replace these measurements. No R26 policy
has been accepted for simulation release or hardware deployment.

Large checkpoints and per-iteration runtime files stay under ignored `runs/`.
The [active run manifest](../config/active_run.json) identifies the local run;
its `status.json` contains the latest reported learning progress.

See the [R26-799 review](Candidate799_Review.md) and [Round 2 plan](Round2_Plan.md).
Round 2 adds per-foot toe-heading measurements and targets 45 mm whole-sole lift.

The [Round 2 final-candidate review](Round2_Candidate1199_Review.md) reports
matched gait measurements, remaining toe-in/clearance limits and video evidence.

The [Round 3 plan](Round3_Plan.md) adds corresponding-phase height consistency,
a schema-aware soft mirror prior and explicit bilateral balance diagnostics.

The [Round 3 review](Round3_Candidate1199_Review.md) records the November 24
iteration decision. The [Round 4 plan](Round4_Plan.md) revises lift gating,
raise-only phase pairing, leg-only mirror loss and lateral bias. The native
Round 2 check reports 9/9 exploratory passes with independent CPU physics.
