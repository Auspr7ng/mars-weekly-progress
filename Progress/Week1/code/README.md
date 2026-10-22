# Week 1 code: two-foot FlexiTac setup

[`flexitac_two_feet_setup.py`](flexitac_two_feet_setup.py) attaches the verified two-foot virtual FlexiTac sensor in Isaac Sim 5.1. It displays live taxel markers and a two-foot heatmap.

## Run

1. Open the local `D:/robotics/flexitac_test.usd` scene in Isaac Sim and stop the timeline.
2. Run the script once from Isaac Sim's Script Editor.
3. Wait for `FlexiTac two-foot setup ready`, then press Play to initialize and update the sensor.
4. Call `builtins.read_flexitac_feet()` for the tactile tensor or `builtins.show_flexitac_pressure_heatmap()` for a saved heatmap.

The scene file and robot assets are not included in this weekly repository. Adapt the hard-coded paths if your local workspace differs. Re-run the script after restarting Isaac Sim; re-running it on the same stage is designed to be safe.

## Interpretation

`read_flexitac_feet()` returns a tensor shaped `(1, 2, 384, 4)`: environment, foot, taxel, then world `x/y/z` and response. The taxel-plane offsets were fitted to one static standing pose. The resulting aggregate-force calibration is provisional and does not validate a pressure map.
