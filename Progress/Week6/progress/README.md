# Week 6 progress - November 18-19, 2026

[Weekly presentation](Weekly_Progress_2026-11-19_Xbox_Arms_Hardware.pptx):
five English slides covering general progress, annotated Xbox controls, the
embedded 67-second simulation recording, current issues and the next test plan.

## Reported gait issues and next tests

The November 19 presentation records two qualitative observations from the
operator: in simulation, changing the arm posture left the robot upright but
both feet appeared stuck; in the lab, foot lift was low and stepping looked
jerky compared with simulation. No measured clearance or timing data was
provided for these observations.

Min's `play_sim_trajectory.py` replays archived simulation targets without live
policy inference. `r25_rollout.py run` performs live R25 inference. The supplied
live command was a supported test with an experimental fixed forward
body-velocity observation of 0.06 m/s and the uncalibrated-accelerometer flag.
The fixed velocity is a candidate contributor, with no confirmed causal link
to the reported jerky gait. The supplied conversation does not identify which
runner produced each observation.

The plan is to iterate training over November 21-22, compare the original and
incoming IMUs next week, and test the updated policy in the lab on Tuesday,
November 24. Reward changes should address foot clearance and smoother steps,
with arm-pose variation included in subsequent training. These are planned
activities rather than completed results.

## Xbox walking, arm and hand control - November 19

Implemented the Week5 simulation extension in Week6: right stick raises/lowers
and spreads/brings back both arms, A opens both hands, Y closes both hands,
and X optionally returns the saved Home. Arm target edits require measured
standstill. Walking is available at any held arm pose using the original
R25-300 policy for the legs; Home readback never gates walking.

The updated interface is English throughout, with the title
`Sunday A1 - Xbox Controller`. Arm speed increased from 0.40 to 0.65 rad/s,
and finger-slider speed from 12 to 24 mm/s. Reset Simulation (or Menu) restores
the robot, policy history and upper-body state after a fall.

Eighteen focused tests passed. The updated scripted full-policy / floating-base
MuJoCo diagnostic passed all 11 checks over 10.34 simulated seconds, including
walking with raised arms, holding the pose while walking, optional Home return,
and recovery after a deliberately induced fall. This is a short empty-hand
software diagnostic; physical Xbox verification and general stability across
all arm poses remain separate. See [results and limits](../results/README.md).

This week is control-program implementation only. No training was launched;
different-arm-pose locomotion training is deferred to next week. See the
[controls](../code/README.md) and [design note](../Arm_Control_Plan.md).

The sections below preserve the earlier Jetson bench record.

## Jetson connection and R25 package

The Mac recognized the robot's Jetson Orin Nano Developer Kit as NVIDIA
`Linux for Tegra` over USB-C. USB Ethernet assigned the Mac `192.168.55.100`
and the Jetson `192.168.55.1`; ping and SSH succeeded. The Jetson runs Ubuntu
22.04.5 LTS, kernel `5.15.148-tegra`, on aarch64. Its existing rollout folder is
`/home/sunday/Sunday_v13_Jetson_Rollout`.

The existing `R25_VERIFIED_SHA256SUMS.txt` check passed for the deployed
contract, R25 runner, actor parameters, and supporting files. The runner's
status reported a 460-input/22-output actor and source checkpoint SHA-256
`3bf72dd99c3c75049db74b4d50c94a6f3ce270af6887a7c048a16130b1ba6202`,
matching the [Week 4 accepted policy](../../Week4/policies/joystick_walking_r25_300/README.md).
The current contract uses a user-captured 22-joint Home candidate, but its
`physical_home_pose_validation_pending` flag remains true. The Jetson contains
earlier Home-preview and motor-run logs for this contract; those records do not
establish ground-walking acceptance.

Mac VS Code's Remote SSH extension opened the Jetson deployment folder with
`SSH: 192.168.55.1` in the window. The [Week 6 guide](../README.md) records
the one-command way to reopen it. VS Code prompted for SSH authentication;
the password was not saved in this repository.

## No-motor experiment

The first R25 shadow attempt stopped because BNO055 registers did not match
the saved configuration. Running the existing `imu-init` command set the
configured IMU mode and axis registers without opening motor ports. A second
attempt stopped because accelerometer calibration was below the required
level 3. The robot was initially lying down; after it was stood up, the IMU
reported approximately 12 degrees of torso tilt, gyro calibration 3, and
accelerometer calibration 0.

With the existing runner's explicit uncalibrated-accelerometer exception,
two supported, stationary diagnostics completed:

| Input | Duration | Samples | Motor output | Candidate-goal errors |
| --- | ---: | ---: | --- | ---: |
| Stop command, zero body-velocity input | 2 s | 100 | Off for every sample | 0 |
| Forward command 0.06 m/s, assumed body velocity 0.06 m/s | 2 s | 100 | Off for every sample | 0 |

The fixed 0.06 m/s body velocity is an experimental observation assumption,
not a measurement. A read-only joint monitor showed no enabled motor torque;
the right hip encoder changed about 30 ticks while the robot was being held or
settling. In the stop-command shadow snapshot, the left elbow was 411 ticks
from its saved Home, the largest of the 22 Home offsets. The existing runner
would move all 22 joints toward Home before policy motor output.

The physical setup at the end of this session had both feet on the floor with
a person holding the torso. The deployed runner is designated for a supported,
suspended, stationary-base bench test. No motor-output command was issued in
this session.

## Xbox input implementation

The new [Jetson Xbox shadow program](../code/jetson_xbox_shadow.py) follows the
[Week 5 left-stick control](../../Week5/code/week4_mapping.py): left stick Y
requests forward/reverse, left stick X requests yaw, and the same dead zone and
circular stick clamp are applied. Its default limits are the smaller R25 bench
commands: +0.06/-0.04 m/s and ±0.15 rad/s. LB is required for a nonzero
command; releasing LB, pressing B, disconnecting, or losing input returns zero
and requires a neutral re-arm when appropriate.

The Jetson Python environment lacks `pygame` and `evdev`, so the program uses
Python standard-library Linux input ioctls. It provides `scan`, `monitor`, and
`shadow` modes, with no motor-output mode. It was copied to the Jetson beside
the existing R25 runner without replacing any deployed R25 file. File hashes
matched after transfer. Five safety and mapping tests passed on both Mac and
Jetson. The Xbox controller had not yet been powered or attached, so its actual
device name, button codes, input permissions, and live stick behavior remain
to be checked. The SSH user could not read an existing unrelated input event
node; only the Xbox event node should receive access if needed.

## Next lab checks

1. Attach and power the Xbox controller on the Jetson. Run `scan` and
   `monitor`, verify left-stick signs, LB, B, neutral re-arm, and disconnect.
2. Run a short Xbox-driven R25 `shadow` session, inspect input and policy logs,
   and confirm every row has `motor_output=False`.
3. Before any motor-output test, use a stable mechanical fixture with both
   feet clear of the floor, keep the motor power cutoff accessible, and
   visually verify the saved Home pose. Resolve the uncalibrated IMU and
   assumed body-velocity limitations before interpreting motion as walking.

No account password or SSH key is stored in this repository.

## Suspended test preflight — November 18, 2026

With the robot suspended, the deployed R25 file checksum check passed again.
A one-second stop `shadow` produced 50 policy samples, and a two-second forward
`shadow` with the existing fixed 0.06 m/s body-velocity assumption produced 100
samples. Both logged `motor_output=False` for every sample and zero candidate
goal errors. The BNO055 gyro and accelerometer calibration values were both 3
throughout the forward run. Readback showed torque off on all 22 motors.

The suspended pose was still far from the saved Home: right ankle roll 869
encoder ticks, right knee roll 714, and both elbow pitches about 630 ticks.
The R25 motor runner would first move all joints toward that saved Home before
policy output. A visual Home check and clear movement space are still needed
before treating a motor trial as validated. The forward `shadow` used an
assumed body velocity, not measured velocity.

### Home preview stopped on actuator fault

After the operator confirmed a stable fixture, clear limb space, and access to
the motor power cutoff, an 8-second `home-preview` was started with a 10-second
countdown. This mode enables motors to move to saved Home but does not run the
policy. The initial maximum Home difference was 868 ticks; the planned ramp
was 1.46 seconds. Other joints approached Home, but `r_elbow_pitch` remained
619 ticks away through repeated `HOME_WAITING` messages. The preview was
interrupted before reaching `HOME_READY` or its hold phase. The shutdown path
reported `TORQUE OFF ERROR: Torque command failed for IDs [4]`; that message
does not identify which arm board returned the error. The run therefore did
**not** verify that all motors were torque off. The operator was told to cut
motor power immediately. Log:
`/home/sunday/Sunday_v13_Jetson_Rollout/r25_home-preview_1795026179136137764.jsonl`.

The operator then reported that only Jetson power remained connected. A
read-only follow-up found no R25 rollout process running; the three USB serial
boards were still enumerated. In the failed preview log, right-arm ID 4 stayed
at 2297 ticks in every saved Home-waiting row, while left-arm ID 4 changed
from 2626 to 2685 ticks. A prior policy log showed right-arm ID 4 varying from
1610 to 1723 ticks, so this motor had moved in an earlier session. The exact
cause of the current failure is not established by these logs.
The operator observed that the right elbow did not move during Home preview,
with no visible mechanical obstruction or loose connector while motor power
was off. A targeted, read-only communication check is the next diagnostic.

Do not start the policy motor run until power-off is confirmed and the right
elbow fault plus the failed torque-off acknowledgement are investigated.

## Min's published rollout package

Reviewed [MoongiMin/sundayA1-Rollout](https://github.com/MoongiMin/sundayA1-Rollout)
on November 18. It packages the R25 actor, contract, legacy `rollout.py`
transport, offline verification, mapping evidence, and documentation. Its
`r25_rollout.py` is the live R25 entry point; `rollout.py` supplies IMU and
servo transport. The published contract SHA-256 is
`300cf9bf8f25b0d18dcb7d35a090198f9c77a05ec7514ec2290ccc5011b3a1d1`,
matching the contract hash recorded in the Jetson's verified mapping document.
The published README identifies the same source checkpoint SHA-256 and saved
November 11 Home capture. A fresh byte-for-byte comparison with Jetson files
was deferred because the Mac could no longer reach the Jetson over SSH.

The published runner's Home wait has no timeout. Its group goal write does
not acknowledge individual servos, and a torque-off failure reports an ID
without the board. Those limitations matter for the right-arm ID 4 failure
observed above. The package does not include a live Xbox input entry point;
its CLI uses fixed commands such as `forward` and `turn-left`.
