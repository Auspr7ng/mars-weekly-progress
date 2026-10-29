# Week 3 - Sunday A1 tactile simulation in mjlab

This week moves our earlier hand and foot tactile work from Isaac Sim to mjlab. The current demo combines two independently controlled grippers, two free balls, four finger tactile arrays, and four foot tactile arrays in one scene.

## Open the demo

**Windows: double-click `start_demo.bat`.**

Use an existing mjlab Python environment. No Isaac Sim installation is needed. If the launcher cannot locate the environment, run `code/configure_python.bat` and enter the full path to that environment's `python.exe`.

From an activated environment:

```powershell
python code/run_demo.py
```

On Linux:

```bash
bash code/start_demo.sh
```

The tested environment uses Python 3.11, mjlab 1.4.0, MuJoCo 3.8.1, MuJoCo Warp 3.8.1, Warp 1.13.0, and CUDA-enabled PyTorch 2.7.1. An NVIDIA GPU and a working driver are required. Package pins are listed in `code/requirements.txt`. The launcher checks compatibility and does not download or change packages automatically. Initial Warp compilation may take a few minutes; wait for `Dual ball grasp ready` and open the URL printed in the terminal.

## Controls

- **Reset left/right ball / and grasp:** reset only the selected ball and gripper, then close that gripper.
- **Release left/right ball:** open the selected gripper and let the ball fall under gravity.
- **Ball position:** adjust each ball's hand-relative XYZ offset before resetting it.
- **Gripper sliders:** control left and right opening independently.
- **Foot tilt / Flatten feet:** change the ankle position targets or restore zero-degree targets.
- **Save:** export finger and foot heatmaps, raw NPZ arrays, and JSON summaries to `results/ball_grasp`.

The browser is the display and control interface. Local Python and mjlab/MuJoCo Warp perform the physical simulation. Tactile values are normalized distance responses, not calibrated pressure. The robot has a fixed base; the demo does not include autonomous reaching, walking, or a trained grasp policy.

The right-finger sampling planes now align with the actual ball-facing contact surfaces. The latest corrected default-grasp heatmap is `results/ball_grasp/dual_held_heatmaps.png`. The updated hand-demo recording embedded on slide 12 shows the corrected implementation. Earlier static screenshots remain historical evidence.

## Weekly deliverables

| Item | Contents |
| --- | --- |
| `start_demo.bat` | Main Windows entry point |
| `code/` | Demo source, launch helpers, dependency pins, and model resources |
| `results/` | Saved heatmaps, raw readings, videos, and validation reports |
| `01_dual_hand_grasp.jpg` | Two-ball grasp interface |
| `02_hand_and_foot_sensor_view.jpg` | Robot and sole sensor view |
| `03_foot_heatmaps.png` | Four foot heatmaps from the saved flat-standing check |
| `04_left_hand_grasp_demo.mp4` | Earlier left-hand grasp/release recording; not a dual-hand video |
| `05_python_architecture.png` | English diagram of the Python control and sensing pipeline |
| `Week3_Migration_Report.md` | Migration findings, sensor definitions, file roles, and validation evidence |
| `Weekly_Progress_2026-10-29_mjlab_Tactile.pptx` | English weekly report with equations, editable response curves, and two embedded screen recordings |

The PowerPoint embeds both original H.264/AAC recordings. In PowerPoint slide show mode, click the video on slide 3 or slide 12 to play. Speaker notes explain the formulas, numerical examples, and current modeling limits.

The model is included as `code/model/sunday_a1.urdf` with its 28 referenced STL files. Runtime MJCF and meshes are in `code/src/sunday_a1_mjlab/assets`. Copy the entire Week3 directory when moving the demo. Selecting a local Python environment creates `code/.local/python_path.txt`; this setting can be changed on another computer.

## Additional commands

```powershell
python code/run_demo.py --check
python code/run_demo.py --regressions
python code/run_demo.py --port 8084
python code/run_demo.py --script preview_robot.py --manual
python code/run_demo.py --script convert_robot.py
```

Windows helpers are in `code`: `check_demo.bat`, `run_regressions.bat`, `configure_python.bat`, and `preview_robot.bat`. The native preview shows the basic model and supports independent L/R gripper opening; the complete two-ball and eight-array interface is provided by `start_demo.bat`.

The conversion command uses the included URDF by default. A replacement model must preserve the expected Sunday A1 joint and link names, and sensor positions must be checked after any geometry change.

## References

- [mjlab framework](https://github.com/mujocolab/mjlab)
- [FlexiTac project](https://flexitac.github.io/)
- [FlexiTac Isaac Sim implementation](https://github.com/binghao-huang/FlexiTac-IsaacSim-Simulation)
- [Vin Robotics mjlab pipeline](https://github.com/VinRobotics/vinrobotics_mjlab)
- [Vin Robotics pipeline article](https://vinrobotics.net/blog/vin-robotics-open-sources-a-reinforcement-learning-training-pipeline-for-humanoid-locomotion)
