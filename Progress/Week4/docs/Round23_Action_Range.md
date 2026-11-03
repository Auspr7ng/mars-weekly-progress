# R23 — Fast-command action authority and collision clearance

R22-200 remained below the fast target: native0.15m/s commanded,0.0953m/s measured. Fast turns reached0.38rad/s but failed hand/knee or foot/foot contact. The8-start diagnostic found ankle policy means up to1.73 before clipping, with approximately10–28% clipping on ankle channels. Hip/knee channels were not clipping. The slow R21-275 baseline remains the published profile.

R23 continues from R22-250. It increases residual target-angle authority by up to50% on ankle roll/pitch, thigh yaw, shoulder roll/pitch and elbow yaw/pitch only when forward demand exceeds0.08m/s or yaw demand exceeds0.12rad/s. Below those thresholds, the mapping is identical. The gain grows linearly and saturates at0.15m/s or0.4rad/s. This adjusts the learned action range; torque limits, joint-speed caps, low-pass target filtering, CAD soft limits, masses/inertias and all collision geometry remain unchanged. GPU and independent native playback use the same formula.

Dense hand/leg clearance margin increases from12mm to18mm. A new signed oriented-box foot/foot separation cost requests8mm separation before a collision. Velocity/yaw tracking and torso damping gain modest weight. Collision and fall terminations and all acceptance gates remain active.

Training samples20% stop,10%0.06m/s anchor,25% varying forward straight,15% reverse,15% walking turns and15% in-place turns. One quarter of non-anchor speed/yaw magnitudes sample the range endpoints explicitly. This replaces R22's only5% random forward-straight allocation and improves coverage at the requested maxima. Commands retain the R22 slew rates.

The initial command/reset/finite-output GPU check and one real PPO update passed before the full run. A separate formula/box-geometry contract confirms old-envelope identity and NumPy/Torch agreement. The final sampler/reward balance is recorded in the formal run's source snapshot. Full candidate checks still require12 command programs×100 starts, independent native feedback, slow/fast gait quality, exact joystick hold/release episodes, and actual video/UI review. No model is accepted on training reward alone.
