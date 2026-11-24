# Round 6 - Vertical gait freedom and stronger leg consistency

This final bounded companion experiment starts from Round 4 checkpoint 400.
Intermediate measurements still show a lower right foot. It tests whether
the inherited 4 mm torso-height reward margin and a weak mirror prior limit
the learned gait, without changing the physical robot model or control limits.

- Parent: Round 4 `model_400.pt`, after 401 updates.
- 512 environments, 400 updates, seed 747, regularizer ramp over 200 updates.
- Preserve the 45 mm trajectory target and 0.20 m/s forward command cap.
- Permit a 10 mm vertical deviation in the torso reward only during active
  gait; the quiet-stop margin stays 4 mm. The roll prior stays unchanged.
  This is a reward tolerance, not root actuation or an imposed body trajectory.
- Leg-only mirror coefficient becomes 0.50. Retain raise-only phase cost -0.8,
  heading/toe-in weights -4.5/-4.0, target-aware lift gating and lateral-bias cost.
- Keep all physical masses, inertia, torque/speed limits, action ranges,
  soft joint limits and failure checks unchanged.

The hypothesis is not established by a static inverse-kinematics search:
that audit fixes root and sole orientation and is only a local search.
Dynamic measured gait determines whether this comparison is useful. Multiple
objective parameters differ from the parent, so it is not a one-factor
causal ablation.

CPU checks explicitly confirm that moving-height tolerance changes while
the roll penalty and standing-height tolerance remain intact. GPU and short
PPO startup checks precede the formal run. Evaluate fixed/random starting
phases, longer forward trials, native physics and the same transition metric
before selecting an experimental checkpoint. The evaluation gates remain
unchanged. No new comparison is planned after this one in the overnight run;
the remaining work is assessment, artifact packaging and GitHub synchronization.

## Recorded completion and recovery

The original process failed while atomically publishing a Windows status file
after update 330. The last complete checkpoint, 325, retained 326 updates.
After adding bounded file-replacement retries and passing the CPU regression
checks, a new run directory resumed 74 updates and saved checkpoint 399.
The retained training path totals 400 updates; five unpersisted updates were
recomputed. Optimizer/normalizers and the saved curriculum counter were restored;
simulator state and RNG were not a bitwise continuation.

See [the failure record](round6_interruption.json),
[resumed startup checks](round6_resume_preflight.json) and
[completed run summary](overnight_training_runs.json). Completion is separate
from the final gait assessment and experimental-candidate selection.
