# Week7 progress - November 19-24, 2026

The overnight comparison retains **Round 4 checkpoint 600** as experimental
progress. Training reward and the newest checkpoint do not determine selection:
longer tests reveal drift regressions in the later Round 4 and Round 6 models.
Read the [final decision](../results/Overnight_Iteration_Review.md),
[measured candidate review](../results/Round4_Checkpoint600_Review.md) and
[portable model bundle](../policies/r26_round4_checkpoint600/README.md).

## Progress against the three requirements

| Requirement | Measured progress | Remaining limit |
| --- | --- | --- |
| Higher feet | Home fast twelve-second mean completed peaks increase from 32.69/26.29 to 36.99/32.58 mm relative to Round 2 | 45 mm reference is unmet; not every step reaches these means |
| Faster forward walking | Command cap increases from the accepted R25 limit of 0.12 to 0.20 m/s; selected Home fast actual speed is 0.1814 m/s | A command cap is not a certified maximum or hardware speed |
| Walking with different arm postures | Seven fixed held poses pass all selected-model matched forward matrices | Continuous arm motion and arbitrary arm configurations are untested |

The selected model passes 84/84 six-second, 168/168 twelve-second, 84/84
random-phase twelve-second and 84/84 twenty-second matched walking trials.
After selection was recorded, another seed-8742 randomized-phase twenty-second
matrix passes 84/84. These finite tests do not establish hardware balance.
Independent CPU MuJoCo checks pass 9/9 cases.

## Completed iterations

| Round | Actual retained training | Result |
| --- | --- | --- |
| 1 | 800 updates | 19/84 short walking passes; low right foot and self-contact remain |
| 2 | 1,200 updates | 84/84 short and 168/168 longer passes; preserved parent |
| 3 | 1,200 updates | Toe-in improves, height regresses; not selected |
| 4 | 800 updates | Keep intermediate checkpoint 600; final 799 drifts in twenty-second trials |
| 5 | 409 of 600 planned updates | Deliberately stop the 35 mm branch after repeated lift regression |
| 6 | 400 updates in retained path | Resume after Windows status-publication failure; final model has heading improvements and more drift |

Round 6 originally attempts 331 updates but only checkpoint 325 is retained.
Seventy-four resumed updates finish iteration 399; five unpersisted updates
are recomputed. The [run summary](../results/overnight_training_runs.json)
records both attempted work and the retained path. The Windows I/O fix is
covered by actual reader-lock and bounded-failure regression checks. The
current checks comprise 18 training invariants and three I/O tests, all passing;
the original Round 6 GPU contract checks pass 20/20.

## Limits and evidence

The strict bilateral diagnostic remains 0 for the selected model. Toe-in,
unequal lift and body roll still need work. All 28 action sequences survive,
and every stop/forward/turn segment passes; reverse is too slow in 28/28,
so complete-sequence passes remain 0/28. The accepted R25 baseline is retained.

Watch the [selected eighteen-second preview](../results/videos/r26_round4_checkpoint600_selected_preview.mp4).
It contains actual Home, raised-arm and raised-and-spread-arm rollouts, with
full-body and leg views. Its trace records 900 physics-control steps without
termination. Open [the monitor](../open_training_monitor.bat) to inspect local
run names, status, progress and curves; the final active run is completed.

## Laboratory context

The current IMU is **ST LSM6DSV16X**. Moon's pinned control repository uses a
historical BNO055 driver and a 460-input R25 actor, while Week7 uses 510 inputs,
22 raw outputs and twelve applied leg actions. It requires an adapted driver
and policy interface, body-speed estimation and verified physical Home alignment.
Read [the hardware reference](../docs/Hardware_Compatibility.md).

The earlier fixed 0.06 m/s velocity observation is distinct from a desired
command; its contribution to jerky real walking remains unisolated. New IMU
fusion alone is not a tested translational-velocity estimator. No hardware
motor run or new IMU register writes were performed by these iterations.
