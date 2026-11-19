# Inputs to the Week 4 walking policy (R25-300)

This describes the **accepted Week 4 policy itself**, using its frozen training
configuration and native observation builder. The actor receives 92 values per
control step with five samples of history (460 values total) and outputs 22
joint-position residuals at 50 Hz.

| Source | Data needed by the actor |
| --- | --- |
| **Every one of the 22 controlled servos** (12 leg, 10 arm/wrist) | Joint angle in radians relative to its saved default position, and joint angular velocity in rad/s. The policy applies a `0.05` scale to angular velocity. All 22 servos have the same requirement. |
| **Five additional modeled joints** (four finger slides, head rotation) | Their position relative to default and velocity also occupy actor observation slots, although the actor does not command them. The finger slides use meters and m/s; head rotation uses radians and rad/s. Week 4's native builder reads their simulated positions and velocities. On hardware, measure them or verify they remain at their defaults before substituting zero relative position and velocity. |
| **Body IMU / orientation estimate** | Body-frame angular velocity `(wx, wy, wz)` in rad/s (scaled by `0.25`); body orientation to calculate body-frame projected gravity `(gx, gy, gz)` and yaw for heading error. The LSM6DSV16X provides a three-axis gyroscope and accelerometer. Its embedded SFLP produces a quaternion and gravity vector. Raw acceleration is not an actor input. |
| **Body linear-velocity estimate** | Body-frame **linear** velocity `(vx, vy, vz)` in m/s. This is an explicit actor input, called `velocity_feedback` in Week 4. Training obtained it directly from simulation. **The LSM6DSV16X datasheet does not list a linear-velocity output**; its "linear acceleration" specification is acceleration, not velocity. Real hardware therefore still needs an external velocity measurement or a validated estimator. Simply integrating acceleration will accumulate bias and drift. |
| **Controller software / joystick** | Desired velocity command `(vx, vy, yaw_rate)`, previous 22-dimensional action, two gait-phase values, and two heading-error values. Phase and heading target are maintained in software; measured orientation is needed to calculate heading error. |

The 22 controlled joints and five additional joints make **27 position and 27
velocity inputs**. The action vector still has only 22 values.

**Foot contact is not required for actor inference.** Week 4 used foot height,
contact state, and force in the *critic* during training, and contact sensors
also informed rewards. They are absent from the deployed actor observation.
Camera, LiDAR, servo current/torque, and temperature are likewise not actor
inputs. Those signals may still be useful for monitoring or other controllers.

LSM6DSV16X reference: ST datasheet DS13510 Rev 3, Section 2.8 (p. 8)
lists SFLP quaternion, gravity vector, and gyro bias. Section 13.30 (p. 139)
lists fusion output rates, including 60, 120 (default), and 240 Hz; an enabled
60 Hz or faster output can cover the policy's 50 Hz control cycle. Integration
must convert sensor axes to body axes, angular-rate units to rad/s, and gravity
to the policy's unit direction vector. The six-axis quaternion's yaw can drift,
so heading should be checked during operation. The actual board wiring,
mounting orientation, and driver output still need to be checked.

Sources: [accepted policy and dimensions](../../Week4/policies/joystick_walking_r25_300/README.md),
[frozen actor/critic configuration](../../Week4/policies/joystick_walking_r25_300/source/env_cfg.py),
[native observation construction](../../Week4/policies/joystick_walking_r25_300/acceptance_source/native_observations.py),
[foot-state definition](../../Week4/policies/joystick_walking_r25_300/source/mdp.py),
and [accepted policy manifest](../../Week4/config/accepted_policy.json).
