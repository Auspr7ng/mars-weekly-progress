# Week 6 Xbox programs

## MuJoCo: Xbox walking, arm and hand control

Run from Week6 with Week4 kept alongside it. This entry point reuses the frozen
Week4 R25-300 checkpoint, model, native physics, actor history and tactile maps.
The Week5 files and Jetson shadow program remain separate.

```powershell
$env:WEEK6_PYTHON = 'D:\robotics\SundayA1_mjlab\.venv\Scripts\python.exe'
& $env:WEEK6_PYTHON -m pip install --target .deps -r code/requirements.txt
.\start_demo.bat
```

Use an existing compatible Week4 Python 3.11 / CUDA environment. The launcher
has the local path above as its fallback; set `WEEK6_PYTHON` on another machine.
The page opens at `http://127.0.0.1:8089/`. `--port 8090` selects another port;
`--no-browser --duration 5` performs a bounded startup check. First launch may
compile GPU kernels. Run in the signed-in desktop session so SDL can see Xbox
Bluetooth/USB input. Installations and generated caches stay in Week6.

| Control | Behavior |
| --- | --- |
| Left stick | Original forward/reverse and turn commands, with Week5 limits |
| Right stick up/down | Raise/lower both arms; lower limit is saved Home |
| Right stick right/left | Spread/bring back both arms; inner limit is Home |
| A / Y | Open / close both hands together; a press latches the aperture goal |
| A and Y together | Hold the current finger target |
| X | Optional smooth return of arms and hands to Home when stopped |
| B / viewer Stop | Zero walking request; hold arm targets and cancel pending hand/Home motion |
| Reset Simulation / Menu, LB released | Restore the robot, policy history, commands and upper-body state, including after a fall |

Centre both sticks and release A/X/Y/Menu and LB to arm after startup, B, a
disconnect, or a 300 ms input timeout. By default no LB is needed for motion;
`--hold-lb` makes LB a hold-to-enable control for walking and arm/hand input.
Neutral right stick holds the arm target. Opening and closing take time;
the four finger sliders move together at 24 mm/s per slider (twice the original
speed). Full open-to-close target travel takes about 1.13 seconds.

You can **stop, pose the arms, and walk directly with that pose**. Walking
never requires Home or a measured arm-position check. Left-stick forward,
reverse and turning commands use the same frozen R25-300 policy and Week5
command limits. During walking, an active arm overlay holds its current targets
while the original policy continues to generate leg actions.

Arm adjustments still require standstill: near-zero filtered walking command,
measured horizontal speed below 0.025 m/s and yaw rate below 0.08 rad/s for
0.60 seconds. Once acquired, small arm-induced sway is tolerated below
0.06 m/s and 0.16 rad/s. These conditions affect arm adjustments only; they
never suppress walking. X returns Home only when stopped, and is optional.

Before the first arm operation, the policy retains its original arm residual
outputs. After posing the arms, the overlay supplies ten arm and four finger
targets after policy target generation. Shoulder speed is 0.65 rad/s (previously
0.40), shared across diagonal stick input. Lift is limited to 0.70 rad above
Home and spread to 0.65 rad beyond Home, with the model's 90% soft position
limits applied. Finger targets span 1.5 to 28.5 mm. Larger slider values open
both hands in the actual model geometry.

The interface uses English text and ASCII separators. Its title is
**Sunday A1 - Xbox Controller**. The **Reset Simulation (Menu)** button stays
available after a fall or contact termination; it restores the starting pose,
clears policy history and arm/hand targets, and resumes the simulation. Center
the sticks and release the action buttons before moving again.

These are target limits, not a verified stable workspace. Start with small
motions. Stationary arm motion changes balance too; the old policy has not
been trained for every new pose. Existing fall/contact checks still pause the
episode. There is no loaded grasp, force controller, or new walking acceptance.
Policy history retains the archived R25 output-action definition while the
upper-body target overlay operates separately; future training must version
that interface consistently with the new reference commands.

See the [control design and next-week notes](../Arm_Control_Plan.md) for staged
standing, fixed-pose walking, and eventually continuous-pose training.

| File | Purpose |
| --- | --- |
| `run_demo.py`, `simulation_player.py` | Hash-checked model/policy loading and native player with explicit arm hooks |
| `native_input.py`, `controller.py` | SDL right/left sticks, A/X/Y/B/Menu, validation, arming and timeout |
| `upper_body.py` | Bilateral targets, rate/position limits, standstill detection for arm edits and walking at any arm pose |
| `viewer.py` | Robot, original eight heatmaps, control state and measured/target trace |
| `week4_mapping.py` | Unchanged Week5 left-stick to Week4 command mapping |
| `test_upper_body.py` | Input failures, actual model geometry/actuation and arbitrary-pose walking and reset tests |
| `check_arm_simulation.py` | Scripted headless policy/physics diagnostic, explicitly separate from real Xbox evidence |

Tests do not need a controller:

```powershell
& $env:WEEK6_PYTHON -B -m unittest discover -s code -p test_upper_body.py -v
& $env:WEEK6_PYTHON -B code/check_arm_simulation.py
```

The second command loads the actual model and policy with an explicitly scripted
input fixture and writes `results/arm_simulation_check.json` plus its
trace. It does not certify Bluetooth/button operation or the full pose envelope.
The diagnostic can run alongside the live viewer and keeps its runtime metadata
in a separate ignored file.
Live input and viewer traces are ignored by Git; the diagnostic summary can be
reviewed and retained as bounded evidence.

### File layout and release contents

Week6 keeps the Windows launcher at the week root, implementation and tests in
`code/`, weekly notes in `progress/`, policy-input notes in `policy_inputs/`,
and the current diagnostic summary/trace in `results/`. Local `.deps/`, `.cache/`
and `.local/` folders, live logs and saved heatmap captures are excluded from Git.
The earlier slower prototype's results are kept only in a local archive.

The portable command is `python -B code/run_demo.py` with a compatible Week4
environment. On Windows, `start_demo.bat` accepts `WEEK6_PYTHON`; the absolute
fallback path is a convenience for the original workstation.

## Jetson Xbox input for R25 (shadow only)

`jetson_xbox_shadow.py` reads an Xbox controller connected directly to the
Jetson and feeds bounded commands to the existing R25 **shadow** runner. It has
no motor-output mode. The original `r25_rollout.py`, contract, and actor files
remain unchanged.

### Jetson controls

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

### On the Jetson

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
