# Week 2 code: four finger FlexiTac sensors

[`sunday_four_finger_flexitac.py`](sunday_four_finger_flexitac.py) installs four independent virtual sensors on the left/right upper/lower Sunday A1 finger links. Each sensor has a 12 × 32 grid (384 taxels), and the script opens a live 2 × 2 heatmap with per-finger statistics.

## Run

1. In Isaac Sim, import the corrected Sunday A1 URDF at `/World/sunday_assembly` and stop the timeline.
2. Run the script in Script Editor. Press Play to initialize the four sensors.
3. Call `builtins.sunday_finger_status()` to print independent readings, or `builtins.sunday_finger_heatmaps()` to save a heatmap image.
4. Use `builtins.set_sunday_grippers(0.03)` to open and `builtins.set_sunday_grippers(0.0)` to close. For the recorded sequence, call `builtins.sunday_demo_cycle(initial_hold_s=3, open_hold_s=4, closed_hold_s=4)`.

The corrected URDF and robot mesh assets are not included here. The script has a local output path under `D:/robotics/SundayA1_FlexiTac/output/`; change it for another computer. Re-running on the same stage updates existing sensors rather than creating duplicates.

## Interpretation

This first contact test queries each sensor against the **opposing finger mesh**. The displayed values are normalized, unsigned SDF proximity/contact proxies. They are **not calibrated pressure or force**. A future object-contact test and calibration are needed before making pressure claims.
