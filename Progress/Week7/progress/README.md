# Week7 progress - November 19, 2026

## Three requirements from the Week6 presentation

1. Increase foot clearance so the robot performs visible, alternating steps.
2. Train forward commands above the released R25 limit of 0.12 m/s.
3. Continue walking while holding different arm postures, instead of freezing
   both feet when the arms leave the training reference.

The implementation retains the verified R25 physical model, then adds arm
references to the actor and critic, masks the policy's arm targets, and trains
leg actions against the changed upper-body configuration. A Home subset helps
retain baseline behavior. Higher-clearance and faster-command targets ramp
gradually rather than changing the viewer limits and assuming success.

## Evidence and interpretation

The laboratory's low foot lift and jerky gait motivate this iteration. The
fixed 0.06 m/s velocity observation in the earlier supported command is not a
confirmed cause. New training uses MuJoCo base velocity. An original/new IMU
comparison and deployment timing/tracking checks remain separate hardware work.

The curriculum direction is consistent with upper-body pose training described
by the [HOMIE authors](https://homietele.github.io/). This is a methodological
reference, not evidence of Sunday A1 performance or a transplanted policy.

## Current stage

The new 510-input task, weight migration, local bounded runner and independent
pose/speed evaluator are implemented. Three CPU tests, nine GPU contract checks
and a two-update PPO startup run passed.

The first full run, `r26_clearance_speed_arms_20261119_01`, started on the local
RTX 4070 Laptop GPU at 20:23 Indianapolis time on November 19. It uses 256
parallel environments, 800 bounded updates, seed 742 and a 600-update target
ramp. Check the [active run manifest](../config/active_run.json) and that run's
`status.json` for current progress. Checkpoints are saved every 25 updates.

An independent 21-case, 84-episode matrix of the transferred initialization
recorded 35 self-contact terminations under the new held-arm interface and
expanded demands. This comparison is scoped to the new task and does not
invalidate or replace the earlier R25 acceptance in its original interface.
It provides a pre-training reference rather than evidence of R26 improvement.

Check `results/` for recorded checks and matrix results. A launched or completed
run is not an accepted new walking policy.

## First completed run and preliminary review

The run completed 800 updates and 6,553,600 transitions in 59.5 minutes, saving
`model_799.pt`. A matched 84-trial, six-second evaluation improved survival
from 49 to 80 trials and the exploratory bilateral walking gate from 8 to 19.
At a 0.20 m/s command with Home arms, mean measured speed increased from
0.119 to 0.173 m/s in the same new-task setup.

The candidate remains unaccepted. Right-foot clearance often stays near
15-20 mm, and raised-and-spread arms at the fast command caused self-contact
in all four trials. See the [review and preview](../results/Candidate799_Review.md)
for per-pose results and the suggested next iteration. That review does not
start another training run.

## Second completed run

Round 2 completed 1,200 updates and 9,830,400 transitions in 94.7 minutes.
At Home and a 0.20 m/s command, inward toe yaw decreased from 15.6/10.1 to
6.3/6.7 degrees, bilateral swing peaks increased from 29.9/19.8 to
32.7/25.3 mm, and mean actual speed increased from 0.173 to 0.181 m/s.
The matched six-second matrix passed the unchanged walking gate in 84/84
trials, compared with 19/84 for the parent. It had no self-contact termination.
The 45 mm target is still not achieved; toe-in persists and Home clearance
remains unequal. See the [Round 2 review](../results/Round2_Candidate1199_Review.md).

A follow-up forward matrix used seed 2742, eight starts per case and 12-second
trials. All 168 trials passed the same walking gate, with no termination.
This is held-pose forward evidence; stop/turn/reverse and hardware remain unvalidated.

## Third iteration started

Round 3 warm-starts from Round 2 `model_1199.pt` with a bounded 1,200-update,
256-environment run, seed 744. It adds corresponding-phase bilateral lift
consistency and a schema-aware soft mirror prior, with varied initial gait
phases and paired asymmetric arm references. Targets remain 45 mm whole-sole
trajectory peak and 0.20 m/s maximum forward command. The new independent
evaluation includes support duty, contact-load proxies, body roll and a
separate bilateral diagnostic. See the [Round 3 plan](../results/Round3_Plan.md).
No Round 3 performance improvement is claimed from startup checks.
