# Xbox input and simulation code

## Files

| File | Role |
| --- | --- |
| `run_demo.py` | Load and hash-check the frozen Week4 checkpoint/model, start native input, and run the native MuJoCo/CPU-policy viewer |
| `native_input.py` | Poll pygame/SDL standard **left-stick X and Y** in one input thread, handle hotplug, log timestamped input |
| `controller.py` | Dead zone, arming, bounded commands, B stop, optional LB enable, 300 ms timeout, reset and ownership state |
| `week4_mapping.py` | Convert the physical left stick into the original Week4 body-frame joystick response |
| `viewer.py` | Robot meshes, original hand/foot distance maps, controls and measured simulation trace |
| `check_controller.py` | Read-only XInput/SDL enumeration and raw/standard control diagnostics |
| `test_controller.py`, `test_week4_mapping.py` | Seventeen input contract and mapping tests |
| `summarize_run.py` | Summarize the latest recorded episode without generating input |

The original Week4 policy and model remain unchanged. The right stick is not bound to walking. Arm/gripper teleoperation has not been implemented.

## Setup and launch

Use the Week4 environment, then install `pygame==2.6.1` into the ignored Week5 `.deps` folder:

```powershell
python -m pip install --target .deps -r code/requirements.txt
python -B code/run_demo.py
python -B code/check_controller.py --seconds 15
```

Run from the Week5 folder. The Windows launchers support `WEEK5_PYTHON`. Input testing needs an ordinary desktop session with the paired controller visible to SDL. No controller is needed for the unit tests:

```powershell
python -B -m unittest discover -s code -p 'test_*.py' -v
```

The viewer uses native MuJoCo physics and CPU actor inference after initialization of the frozen mjlab environment on the GPU. It preserves Week4's command slew rates, action filtering and termination checks. Zero command requests braking and support; Pause freezes simulation time separately.

The native input mode does not use browser focus as its enable condition. The spring-centred left stick and explicit stop controls govern input while the page is connected. A stale reader or disconnected device disarms the command. Disconnect/reconnect handling exists in code but needs a dedicated physical test.

## Source reference

Open Duck Mini Runtime's [Xbox controller](https://github.com/apirrone/Open_Duck_Mini_Runtime/blob/v2/mini_bdx_runtime/mini_bdx_runtime/xbox_controller.py) informed the pygame/SDL input approach. Sunday A1 uses a different mapping: its entire walking command is on the left stick, and it loads its own policy. No Open Duck policy or joint mapping is used.
