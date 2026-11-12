# Week 5 — Xbox control, tactile research, and cart validation

**Weekly progress: November 12, 2026.** Three separate work items:

| Progress | Status | Entry point |
| --- | --- | --- |
| Xbox control of Sunday A1 | Real Bluetooth input drives the simulated robot through the frozen Week4 R25-300 policy | [Controller code](code/README.md) |
| Foot tactile research | Source review and a proposed experiment plan. No validated hardware detector or new tactile policy yet | [Research plan](research/Foot_Tactile_Research_Plan.md) |
| Cart model validation | Original XML fails to compile. A diagnostic copy completed independent physics tests with assumed caster inertials | [Cart report](cart_validation/Validation_Report.md) |

[Weekly presentation](Weekly_Progress_2026-11-12_Xbox_Tactile_Cart.pptx) and [selected media](progress/README.md).

## Xbox walking demo

Keep Week4 and Week5 as sibling folders under `Progress/`. Reuse a compatible Week4 Python 3.11 environment and NVIDIA GPU. Install the Week5 input dependency locally, then launch:

```powershell
$env:WEEK5_PYTHON = 'D:\path\to\python.exe'
& $env:WEEK5_PYTHON -m pip install --target .deps -r code/requirements.txt
.\start_demo.bat
```

The launcher opens `http://127.0.0.1:8087/`. The existing local setup can use the launcher's fallback environment. If that port is already occupied, reopen the running viewer or close it before restarting. `start_demo.bat --port 8088` selects another port.

| Control | Behavior |
| --- | --- |
| Left stick up/down | Forward/reverse relative to the robot |
| Left stick left/right | Turn relative to the robot |
| Left stick diagonal | Walk and turn together |
| Left stick centred | Request zero velocity and let the policy brake |
| B or viewer Stop | Stop, then require centred input before resuming |
| Menu, with LB released | Reset the simulated episode |
| Right stick | Reserved for future arm/gripper input. It has no walking binding |

The left stick replaces Week4's 3D spring joystick and calls the same command mapper after a hardware drift dead zone. Camera orientation does not change the command frame. Default operation does not require LB. `--hold-lb` enables the optional hold-to-move mode. Native input continues in the background while a viewer is connected. Closing all viewer clients requests zero.

Published command limits remain 0.12 m/s forward, 0.06 m/s reverse, and ±0.25 rad/s yaw. These are requested limits. Instantaneous measured velocities can overshoot them. This is a simulation interface, not physical-robot walking acceptance. The policy still controls its established arm reference poses.

## Evidence and limits

- Seventeen command/mapping tests pass, including parity with Week4's joystick mapping and diagonal walk/turn.
- The [interactive snapshot](results/run_verification.json) records real controller input, forward/reverse and both turn directions, with 138 sampled diagonal commands. It is a bounded snapshot of one session, not a new policy acceptance campaign.
- The [raw input](results/input_trace.jsonl) and [native simulation trace](results/simulation_trace.jsonl) preserve the interaction. `summarize_run.py` selects the latest episode in those traces, which can differ from the earlier bounded snapshot as the session continues.
- Hardware B/Menu presses and Bluetooth unplug/reconnect were not independently verified in that snapshot. Unit tests cover command stop/reset and timeout logic separately.
- The first sandboxed process could not see the controller. Normal desktop pygame/SDL detected the Xbox Series X Controller using standard mapping. XInput still reported disconnected. Hardware-facing programs should run in the signed-in desktop session.

## Foot tactile research

The reviewed firmware and viewer both configure 32×32 frames. The hardware viewer subtracts an unloaded baseline and thresholds ADC responses. The simulation's sole maps encode geometric distance response. Both require separate interpretation.

The proposed study first compares foot-contact detectors on labelled hardware data, then matches the observable signal in simulation and compares policies. [The plan](research/Foot_Tactile_Research_Plan.md) marks source facts, hypotheses and pending measurements. The questions in the report originated with this project; they are not a lab response or an established solution.

## Cart validation

The [unchanged cart source](cart_validation/source/cart_v2.xml) has moving caster swivel bodies without their own mass/inertia and fails under MuJoCo 3.8.1. The [diagnostic copy](cart_validation/results/cart_test_fork_0.05kg.xml) adds an assumed 0.05 kg fork per caster, inertias, and a test floor. Those are test assumptions, not measured hardware parameters.

Eleven base cases and two mass-sensitivity cases remained finite without warnings or tipping. The model showed strong rolling resistance and slow caster reorientation. [The report](cart_validation/Validation_Report.md) and [36-second video](cart_validation/results/cart_validation.mp4) document the conditions. Robot-cart pushing remains next week's task.

## Release scope

This release contains English code documentation, original test evidence and the weekly deck. Complete lab reference downloads, other people's weekly decks, private SharePoint links, caches and installed packages remain in the local archive. The original cart XML is included because it is the specific model under test. No shared cloud file was edited.
