# R22 — Hold-to-move joystick and faster locomotion

The user clarified that this should behave like a spring-centred joystick: motion only while holding and dragging; release must recenter and request zero velocity. The previous release-and-hold direction arrow is superseded. All display text stays in English, matching Week3.

## Interaction

- Dragging local +X requests forward velocity; −X requests reverse after the upgraded policy is accepted. Local ±Y directly requests left/right yaw rate.
- The reference frame above the head follows robot yaw. The command is body-relative throttle/steering, without waiting to align to a fixed world direction.
- Mouse release immediately clears the raw command and recentres the handle. A disconnected controlling client also clears it. Physical braking still runs through the learned policy and the same simulated actuators.
- During training, the live page retains the previously verified speed caps. Larger caps are only published with a separately verified policy.

## Training changes

- Forward target range extends to0.15m/s; reverse to−0.06m/s; yaw rate to±0.4rad/s.
- A200-iteration curriculum gradually extends the old command envelope. The0.06m/s forward anchor and zero commands remain in the distribution.
- Commands slew at0.5m/s² and0.8rad/s², compared with0.25 and0.35 previously. Native CPU evaluations use the identical new command rates.
- Reverse arm-swing shaping uses absolute speed for its normalization; foot trajectory signs already follow the signed forward command.
- Physical masses/inertias, joint speed/torque caps, geometry, stance, observation/action dimensions, sensor source files and termination conditions are unchanged.

Training starts from the frozen R21-275 baseline, with an explicit optimizer reset. A32-environment command/reset/finite-output check passed, including command sampling near0.15m/s and0.4rad/s and nonzero reverse commands. A real PPO smoke update must pass before the full run.

The planned run is1024 environments×600 bounded PPO updates. Validation retains the seven old command programs and adds fast straight walking, fast speed/stop/restart, fast walking turns, fast in-place turns and reverse/stop. Fast0.15m/s segments require measured mean forward speed≥0.12m/s;0.4rad/s turns require signed mean yaw rate≥0.28rad/s. Old survival, stepping, drift, quiet stopping and gait-quality gates remain in effect. Both0.06 and0.15m/s receive independent GPU/native20-second gait audits.

The original R21-275 baseline remains archived; its700/700 command-program successes, native checks and old direction-arrow tests do not certify the expanded envelope or the new joystick semantics.
