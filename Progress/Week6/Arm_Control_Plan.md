# Week 6 control design and next-week notes

Updated November 19, 2026. This week implements the MuJoCo Xbox control
program only. No training is launched and the R25-300 checkpoint is unchanged.

## Controls

| Input | Action |
| --- | --- |
| Left stick | Forward, reverse and turn with the original R25-300 walking policy, at any held arm pose |
| Right stick up/down | Raise/lower both arms while stopped |
| Right stick right/left | Spread/bring back both arms while stopped |
| A / Y | Open / close both hands together |
| A + Y | Hold the current finger target |
| X | Optional smooth return to Home while stopped |
| B | Stop commands and hold arm targets |
| Reset Simulation / Menu | Restore the simulated robot and clear policy/arm state, including after a fall |

## Motion and mode handling

Arm target speed is 0.65 rad/s, up from 0.40. Finger-slider speed is 24 mm/s,
up from 12. Full 27 mm open-to-close target travel takes about 1.13 seconds.
The unchanged position envelope is 0.70 rad of lift and 0.65 rad of spread
beyond Home, further bounded by model soft limits.

Standstill detection controls when arm targets may be edited: filtered walking
commands must be near zero, horizontal speed below 0.025 m/s, and yaw rate
below 0.08 rad/s for 0.60 seconds. Hysteresis tolerates small arm-induced sway.
Walking commands always pass through, regardless of arm pose or Home readback.
During walking, the overlay holds the selected arm pose while the original
policy continues to supply the leg targets. No new balance policy is trained.

The original fall/contact checks pause the episode. Reset Simulation restores
the initial robot pose, clears the policy history and walking command, restores
arm/hand Home targets, and resumes simulation. The page uses English text and
ASCII separators to avoid the earlier corrupted separator character.

## Next week

Different arm poses can change balance, so this control interface does not
establish walking stability at every pose. Training is deferred to next week.
Start with a few standing-tested fixed poses and low-speed movement before
considering continuously changing poses and loaded/contact tasks.

R25 already observes whole-body joint positions and velocities. If a future
policy adds arm-reference commands, version the observation/history,
normalization and deployment contract together. The current 460-input actor
cannot directly consume an expanded observation vector.

[HOMIE](https://arxiv.org/abs/2502.13013) and its
[author project page](https://homietele.github.io/) provide a reference for
upper-body pose curricula. These results concern other robots and do not
establish performance for Sunday A1.

[Launch and usage](code/README.md) | [Validation](results/README.md)
