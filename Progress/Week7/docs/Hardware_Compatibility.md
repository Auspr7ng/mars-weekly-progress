# Hardware reference and current IMU

Updated November 24, 2026 from the user's current hardware correction.
**The current IMU is ST LSM6DSV16X, not the historical BNO055.** The Week7
training/evaluation programs remain simulation programs; this document records
the real-robot interface context without claiming a tested hardware port.

## Moon's control repository

Reference: [MoongiMin/sundayA1-Rollout](https://github.com/MoongiMin/sundayA1-Rollout).
Reviewed default-branch commit:
[`cd9da497c1cf9030bc2f107bf5bf6e648ce297a3`](https://github.com/MoongiMin/sundayA1-Rollout/commit/cd9da497c1cf9030bc2f107bf5bf6e648ce297a3).
The inspected code is pinned below so later changes are distinguishable.

- [`r25_rollout.py`](https://github.com/MoongiMin/sundayA1-Rollout/blob/cd9da497c1cf9030bc2f107bf5bf6e648ce297a3/r25_rollout.py)
  performs live NumPy R25 actor inference, encoder conversion and target
  shaping. It imports `BNO055` from `rollout.py` and explicitly expects its
  verified bus/address configuration (bus 7, address 0x28). That driver/config
  is historical and does not support the user's new IMU.
- [`play_sim_trajectory.py`](https://github.com/MoongiMin/sundayA1-Rollout/blob/cd9da497c1cf9030bc2f107bf5bf6e648ce297a3/play_sim_trajectory.py)
  replays archived targets without live policy or IMU feedback. It is a
  different experiment from live policy inference.
- [`r25_contract.json`](https://github.com/MoongiMin/sundayA1-Rollout/blob/cd9da497c1cf9030bc2f107bf5bf6e648ce297a3/r25_contract.json)
  is the mapping/angle/action contract. Addresses are `(board, motor ID)`;
  arm-board IDs are reused across sides. Preserve the actual IDs, signs,
  captured Home ticks and model references. Do not reapply historical sign
  corrections or stretch hand-recorded tick ranges into model angle limits.
- [`POLICY_IO.md`](https://github.com/MoongiMin/sundayA1-Rollout/blob/cd9da497c1cf9030bc2f107bf5bf6e648ce297a3/docs/POLICY_IO.md)
  documents term-major history, scaling and target handling.

The source R25 checkpoint SHA-256 matches this repository's accepted R25-300:
`3bf72dd99c3c75049db74b4d50c94a6f3ce270af6887a7c048a16130b1ba6202`.
The frozen model and Home hashes also match the Week7 parent dynamics. This
establishes common software provenance, not physical Home alignment or balance.

## Policy interface differences

| Property | Pinned Moon R25 runner | Week7 R26 simulation |
| --- | --- | --- |
| Actor inputs | 460 | 510 |
| History | Five samples per term | Same prefix plus five samples of ten arm references |
| Raw actor outputs | 22 | 22 |
| Applied policy outputs | 22 joint targets | 12 leg targets; held reference owns ten arm targets |
| Control timestep | Nominal 20 ms | 20 ms |
| Body velocity input | Zero or experimental fixed X value | Measured simulated body velocity |
| Forward command tested here | Separate from fixed velocity observation | Up to 0.20 m/s |
| Cadence | Archived R25 command oscillator | Additional shortening above 0.15 m/s |

The pinned actor loader asserts a `(256,460)` first layer. Copying a Week7
checkpoint/weight file into that loader is incompatible. A future port must
export the new actor/normalizer, append references in the exact term-major
order, retain 22 raw-action history values, override arm outputs with references,
and match the updated cadence and actuator target handling.

The previously used `--experimental-fixed-body-vx 0.06` supplies a synthetic
body-X velocity observation. It does not measure speed and is distinct from
the desired forward command. Its role in the observed jerky lab gait has not
been isolated. A new IMU does not automatically supply drift-free translational
velocity. Changing that fixed value is not a validated velocity estimator.

## New LSM6DSV16X reference

Source: the user-supplied ST datasheet `lsm6dsv16x.pdf`, DS13510 Rev 3,
March 2023, 198 pages. The source PDF is not redistributed in this repository.
Manufacturer product page: [LSM6DSV16X](https://www.st.com/en/mems-and-sensors/lsm6dsv16x.html).

- Page 1: six-axis accelerometer/gyroscope with SPI, I2C and MIPI I3C
  interfaces; embedded sensor fusion is available.
- Page 8, section 2.8: the SFLP block can produce a game rotation quaternion,
  gravity vector and gyro-bias vector from accelerometer/gyroscope processing.
  Yaw performance is expressed as drift over time; do not treat it as an
  absolute magnetometer heading.
- Page 12, Table 3: gyro sensitivity depends on full scale (4.375 through
  140 mdps/LSB); accelerometer sensitivity likewise depends on full scale
  (0.061 through 0.488 mg/LSB). These are not BNO055 register scales.
- Page 64, section 9.13: `WHO_AM_I` is register `0x0F`, expected value `0x70`.
- Page 110, Table 218: SFLP FIFO tags include `0x13` game rotation,
  `0x16` gyro bias and `0x17` gravity. Actual FIFO decoding must use its
  documented formats rather than treating all tagged words as raw gyro.
- Page 139, section 13.30: SFLP output-rate options are 15, 30, 60, 120,
  240 and 480 Hz. Sensor rate and the 50 Hz policy loop are separate choices.

The actual breakout/module, connection interface, Jetson bus/address and
mounting rotation are not yet recorded. Do not assume the old bus 7/0x28
configuration or identity mount matrix is valid for this new installation.

## Required normalized observation boundary

A future hardware adapter must produce fresh, timestamped measurements in
the robot body frame: angular velocity in rad/s, normalized projected gravity
(approximately `[0,0,-1]` when upright in the model), and relative yaw in the
runner's convention. Sensor FIFO layout/quaternion conventions, mount rotation,
bias handling, timestamps, filtering and sample age must be checked explicitly.
The old BNO055 calibration-level flags are not an LSM6DSV16X health interface.

SFLP provides an attitude/gravity source, not a complete replacement for the
policy's body-linear-velocity observation. Estimator behavior, physical Home
alignment, encoder mapping and motor tracking remain separate from IMU fusion.
The preserved repository context is a reference for that work; no new IMU
register writes or motor run were performed by the Week7 training task.
