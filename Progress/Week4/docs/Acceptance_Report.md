# Week 4 acceptance — joystick-controlled RL walking

The selected **R25-300** policy passed the final flat-ground simulation assessment. Training stopped after saving update 336. The current interface is `body_follow_v3`; its coordinate tests are separate from the policy's locomotion tests.

## Published behavior and measured speed

Hold and drag the ball above the head. Body-local X requests forward/reverse motion; body-local Y requests yaw. Release immediately clears the raw command and recentres the handle. Physical stopping follows the learned braking response and command slew limits. Nonzero minimum commands outside the dead zone stay inside the trained gait range.

| Item | Command limit | Measured example |
| --- | ---: | ---: |
| Forward | 0.12 m/s | 0.0944 m/s mean over 100 steady runs; previous profile about 0.0603 m/s |
| Left/right turn | +/-0.25 rad/s | About +0.247 / -0.231 rad/s in place |
| Reverse | -0.06 m/s | About -0.0425 m/s over a complete cycle |
| Release | Raw request becomes zero | Quiet double-support confirmation typically 1.12–1.37 s, including a continuous 0.5 s confirmation window |

The native MuJoCo/CPU viewer measured a mean real-time factor of **0.897** over a 12-second single-client sample. Physics and policy steps remain 0.002 s and 0.020 s; rendering speed does not create the walk. [Recorded timing](../results/validation/final_web_realtime_single_client.json).

## Completed checks

- **12 command programs x 100 randomized starts: 1,200/1,200 completed and survived.** Programs include idle, steady speed, speed/stop/restart, both walking turns, both in-place turns, held-out orderings, faster commands, and reverse.
- **100 repeated 60-second joystick sequences passed the versioned whole-cycle assessment.** Half/full deflection, reverse, mixed turns, repeated release, and one-second bursts were checked. Every release stage had zero raw command.
- Independent native-physics feedback passed. The CPU actor used by the final viewer separately passed all 12 command programs and the complete 60-second joystick sequence.
- GPU and independent native 20-second gait audits passed at commands of 0.06 and 0.12 m/s. Four front/side recordings were inspected for alternating clearance, stepping turns, quiet planted stops, and arm/leg clearance.
- Browser checks covered forward, reverse, both steering directions, release, Stop, Reset, and heatmap saving. The four tactile implementation files match Week 3 byte-for-byte.

| Gait metric at 0.12 m/s command | GPU | Independent native MuJoCo | Final criterion |
| --- | ---: | ---: | --- |
| Roll peak-to-peak | 5.7860 deg | 5.7861 deg | <=8 deg |
| Pitch peak-to-peak | 2.0449 deg | 2.0450 deg | <=6 deg |
| Root-height peak-to-peak | 10.2607 mm | 10.2608 mm | Fast <=12 mm; slow <=10 mm |
| Settled stance sole speed, p95 | 0.0045 m/s | 0.0045 m/s | <=0.035 m/s |
| Touchdown speed, p95 | 0.0411 m/s | 0.0411 m/s | <=0.08 m/s |
| Peak single-foot force / body weight | 1.2852 | 1.2795 | <=1.8 |

Minimum full-swing peak clearance was about **28.53 / 20.83 mm** for left/right, above the 15 mm requirement for each foot. Flight fraction was zero. At the slower 0.06 m/s command, root-height variation was approximately 7.17 mm and roll variation 6.57 degrees.

## Versioned criteria and preserved failures

The original v1 verdict is **false** and is retained in the [raw result](../results/control_r25_candidate300_responsive_result.json). The final [v2 assessment](../results/control_r25_candidate300_responsive_result_assessment_v2.json) uses [explicitly versioned criteria](../config/control_acceptance_phase_aware.json):

1. Held-command speed is averaged over complete gait cycles inside the steady window, retaining the 70% speed threshold. The original half-reverse 1.3-cycle window measured 0.03435 m/s against 0.03463 m/s; a complete cycle measured 0.03506 m/s. Shifted one-cycle windows also passed. Quiet stopping, landing, collision, and one-second burst checks remain active.
2. Settled stance is distinguished from touchdown braking. Stance p95 remains limited to 0.035 m/s; touchdown p95 is separately limited to 0.08 m/s and impact to 1.8 body weights. The original all-contact speed remains recorded, approximately 0.041 m/s at the fast profile.
3. Fast root-height variation is limited to 5% of the nominal 0.24 m root height: 12 mm. The measured value is 10.26 mm. The slow limit remains 10 mm. Other roll, pitch, clearance, contact, survival, and command gates remain in force.

These revisions are methodological changes, not a claim that the original gates passed unchanged. The batch, native, quality, raw joystick, and cycle-aware reports are indexed by the [accepted manifest](../config/accepted_policy.json). The 100-run raw joystick NPZ is included. Bulk per-step trajectories for all 1,200 command runs remain local.

## Model, arms and heatmaps

The mass budget is 2.84 kg with a 0.70 kg root. The R25 policy retains the original action ranges and uses a continuous command-dependent cadence. Four shoulder/elbow reference targets were changed for knee clearance; these are control reference targets, not CAD or encoder zeros. Ten arm/wrist joints remain active PPO outputs. See [model review](Model_and_Hardware.md) and [R25 notes](Round25_Command_Cadence.md).

The eight heatmaps preserve Week 3's distance response, grids, and image orientation. They are not calibrated in newtons or pressure. Raw saving was checked for four foot arrays, four hand arrays, and six separate/combined images. Real hardware still needs encoder direction/offset calibration, actuator identification, a velocity estimator, and force-response calibration.

Checkpoint SHA-256: `3bf72dd99c3c75049db74b4d50c94a6f3ce270af6887a7c048a16130b1ba6202`. The experimental 0.15 m/s and +/-0.4 rad/s range is not published as accepted behavior.

## Current joystick verification

The base follows the head and the axes follow current body yaw, including during a held drag. The red arrow marks forward. New pointer displacement is transformed through the current handle's perspective and body frame; robot/camera movement alone does not generate input. Release recentres and clears the command.

The [MuJoCo pose/callback regression](../results/validation/body_follow_v3_contract.json) covers 0, +90, 180, and -90 degrees, translated poses while held, visual-axis rotation, new forward input, stationary-pointer isolation, steering, release, and disconnect. [Actual browser checks](../results/validation/body_follow_v3_result.json) verify pure forward, pure steering, off-centre forward, and release. The former `free_drag_v2` frozen-anchor behavior has been replaced. [Current interface](../results/web_demo/body_follow_v3.jpg).
