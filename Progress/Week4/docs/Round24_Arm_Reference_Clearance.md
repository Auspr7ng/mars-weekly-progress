# R24 — Safe reference arms and gradual ankle authority

R23-100 still failed fast starts on hand/knee contact, and fast turns on foot/foot contact or body/ground. Its checkpoints and diagnostics remain archived. Increasing action authority on many joints at once changed a saturated policy too abruptly.

Kinematic checks show that shoulder pitch+0.15rad moves each gripper outward by approximately27mm; elbow pitch+0.3rad raises and advances it. The new reference changes both shoulder-pitch targets from0 to0.15rad and both elbow-pitch targets from0.7 to1.0rad. All other reference fields remain identical. These are neutral control targets, not CAD or encoder zeros. All four targets lie inside the existing90% soft limits. A5-second native PD check with a ground plane passed: minimum root height0.2392m and maximum roll/pitch0.0251rad.

An explicit weights-only transfer from R22-275 validates the exact four reference changes, physical model hash, source/target nominal hashes and unchanged observation dimensions. In initial native policy tests, no hand/leg collision occurred in the seven inspected programs. Fast failures instead involved the two feet, which motivates continued landing/clearance training. That initial result is not acceptance.

R24 retains R23's endpoint-balanced command sampling, tracking, hand clearance and foot separation costs, with stronger torso damping and foot clearance weights. Extra residual authority applies only to ankle roll/pitch for forward speeds above0.08m/s; it no longer expands arms or thigh yaw or reacts to pure yaw commands. During training only, authority increases smoothly to the full factor over100 updates. Evaluation always uses the final mapping. Arms remain10 active PPO joints and may swing within their original ranges around the safer reference.

Torque/speed limits, target filtering, physical joint limits, body masses/inertias, collision geometry, sensors and termination gates are unchanged. Acceptance still requires full command/stop/turn/reverse and hold/release evaluations, slow/fast gait checks, video review and the English web UI. The live demo keeps the accepted R21-275 profile until those checks pass.
