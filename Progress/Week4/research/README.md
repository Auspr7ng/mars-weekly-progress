# Research inputs and derived audits

The initial robot came from the Week 3 Sunday A1 URDF and 28 meshes. This folder publishes derived technical audits and joint tables; original lab Word/CAD source files and extracted personal/cloud metadata remain local.

| Reference | Role in this work |
| --- | --- |
| [mjlab](https://github.com/mujocolab/mjlab) | Manager-based environment, GPU MuJoCo-Warp simulation, PPO integration, observation/contact structure, and velocity-task patterns |
| [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) | End-to-end hardware/model/actuator/training workflow; its weights and hardware constants were not transplanted |
| [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) | Command-conditioned locomotion, timing and motor/latency modeling ideas |
| [Open Duck reference-motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) | A model-specific feasibility/reference option when unconstrained rewards produced scuffing |
| [Vibe Robotics SDK](https://github.com/viberobotics/vibe_robotics_sdk) | Parent-platform topology and provisional actuator/body template information; later measured modified-robot metadata takes precedence |
| [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | Robot-package separation and velocity-control task structure |
| [VinRobotics mjlab](https://github.com/VinRobotics/vinrobotics_mjlab) | Reproducible robot/task packaging and evaluation patterns |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO, history/normalization handling and mirror consistency |

These references informed implementation choices; their geometry, initial joint values, trained policies, actuator limits and acceptance numbers were not assumed valid for Sunday A1.

The [Week 3 model audit](week3_model_audit.json) describes the initial, superseded URDF mass declarations. The [SDK comparison](vibe_sdk_model_comparison.json) is also historical. Current mass/dynamics conclusions are in [Model_and_Hardware.md](../docs/Model_and_Hardware.md).

Native CAD-derived evidence:

- [Joint limits, axes and saved zeros](hardware_metadata/2026-11-01/fusion_joint_limits_axes_zero.csv).
- [Link mass/COM/inertia properties](hardware_metadata/2026-11-01/fusion_link_mass_properties.csv). Steel-derived values are geometric audit values, not accepted measured masses. The material label is translated to English in this copy.
- [CAD/URDF comparison](hardware_metadata/2026-11-01/fusion_verified_comparison.json).
- [Initial joint inventory](joint_inventory.csv).

The final model's root mass, voltage/model discrepancies and calibration gaps remain explicit. No additional manufacturer torque/speed rating was inferred from an inaccessible product page.
