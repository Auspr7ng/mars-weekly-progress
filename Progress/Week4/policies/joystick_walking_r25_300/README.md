# Accepted R25-300 policy

This is the frozen flat-ground simulation policy selected after the complete command, gait, native-feedback and visual review. `model_300.pt` is preserved byte-for-byte. `run.json`, `source`, `dynamics`, `agent.yaml` and `env.yaml` record its training provenance and interfaces.

- Actor: 460 normalized/history features; critic: 98 features.
- Action: 22 position residuals, 12 leg plus 10 arm/wrist joints.
- Physics/control: 500 Hz / 50 Hz.
- Published limits: 0.12 m/s forward, -0.06 m/s reverse, +/-0.25 rad/s yaw.
- A simulated body-velocity observation is used; hardware needs an estimator.

Use the [main launcher](../../README.md) so the archived model/reference, filtering, limits, cadence and observation order remain consistent. The current manifest is [config/accepted_policy.json](../../config/accepted_policy.json). `acceptance.json` is the earlier acceptance snapshot; the current viewer's `body_follow_v3` checks are linked by the main manifest and [report](../../docs/Acceptance_Report.md).

Intermediate ancestry checkpoints remain local. Metadata references to those paths are historical records, not additional files required for inference. The frozen source hashes in `run.json` are verified by `python code/run_demo.py --check` from the Week4 directory.
