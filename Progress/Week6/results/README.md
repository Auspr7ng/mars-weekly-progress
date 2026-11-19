# Xbox arm-control checks - November 19, 2026

No training was launched and the R25-300 checkpoint is unchanged.

## Current controls

- `test_upper_body.py`: 18 tests passed, covering input failures, geometry,
  actual actuator tracking, faster motion timing, arm adjustment at standstill,
  forward/reverse/turn commands at non-Home poses, held arm targets and reset.
- [Current full-loop summary](arm_simulation_check.json): all 11 checks passed
  over 10.34 simulated seconds, with 259 samples. The real archived policy and
  floating-base MuJoCo physics raised/spread both arms, opened/closed both
  hands, returned Home optionally, and walked with a non-Home arm pose.
- The fixture deliberately tilted the robot to 85 degrees, triggered the
  existing episode-end condition, requested the same reset flag as the viewer
  button, and verified the initial pose/arm state and continued simulation.
  The recorded maximum tilt includes this injected fall.
- [Current trace](arm_simulation_trace.jsonl) identifies episode boundaries,
  measured joints, target joints, accepted commands, mode and tilt.

Inputs are explicitly scripted, not physical Xbox evidence. These checks do
not establish stability across the full pose envelope or loaded grasping.
The interface uses English labels and ASCII separators. Full target travel is
about 1.08 seconds for 0.70 rad arm lift and 1.13 seconds for finger closing.

See the [code guide](../code/README.md) to repeat the current checks. Live
viewer/input logs, local installs and caches are ignored by Git.
