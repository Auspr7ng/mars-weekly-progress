# Jetson Xbox input for R25

`jetson_xbox_shadow.py` reads an Xbox controller connected directly to the
Jetson and feeds bounded commands to the existing R25 **shadow** runner. It has
no motor-output mode. The original `r25_rollout.py`, contract, and actor files
remain unchanged.

## Controls

- Left stick Y: forward/reverse, capped at +0.06 / -0.04 m/s.
- Left stick X: yaw, capped at ±0.15 rad/s.
- Hold LB to issue a nonzero command. Release LB for zero; centre the stick
  before re-arming.
- B stops and requires LB release plus a centred stick before re-arming.
- Disconnect, unreadable device, or program exit requests zero.

The mapping follows the Week 5 simulation controller's dead zone and circular
stick limit, with the smaller command envelope already used by the Jetson R25
bench script. `shadow` computes policy outputs and candidate motor targets but
does not enable torque or send motor goals. Its fixed body-velocity option is an
assumption, not a measurement.

## On the Jetson

Place this file beside `r25_rollout.py` in `~/Sunday_v13_Jetson_Rollout`.
After connecting the Xbox controller, run from that directory:

```bash
.venv/bin/python jetson_xbox_shadow.py scan
.venv/bin/python jetson_xbox_shadow.py monitor --seconds 15
.venv/bin/python jetson_xbox_shadow.py shadow --seconds 5 --allow-uncalibrated-accel
```

`scan` shows the Linux event device and whether `sunday` can read it. If it is
unreadable, give `sunday` read access to that specific Xbox event device before
running `monitor`; do not grant access to keyboard or other input devices.
`monitor` shows axes, LB, B, arming state, and requested velocity without
opening any robot hardware. Confirm the physical stick and buttons match those
readings before `shadow`.

The `shadow` mode uses the R25 deployment's existing IMU, motor board checks,
policy actor, and JSONL logging. It also writes an `xbox_input_*.jsonl` log
beside this script. It requires a near-upright torso and may need the explicit
`--allow-uncalibrated-accel` flag while the robot is held for a stationary
diagnostic. That flag does not make the IMU calibrated or authorize a motor run.
