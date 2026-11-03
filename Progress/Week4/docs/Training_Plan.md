# Sunday A1 locomotion training plan

This is an English consolidation of the initial Week 4 plan and its implementation decisions. Initial proposal values below are not the final acceptance criteria. The released configuration, selected weights, and achieved results are specified in [Acceptance_Report.md](Acceptance_Report.md), [Training_History.md](Training_History.md), and the frozen policy source.

## Objective and priorities

Train the complete Week 3 Sunday A1 robot to stand, start, walk, change speed, stop, restart, and turn on flat ground using MuJoCo/mjlab and PPO. Preserve both arms, wrists, grippers, head, and their dynamics. Walking quality takes priority over the interactive control interface.

Keep four foot and four hand distance heatmaps visible during playback, with the original Week 3 response definitions. Compare a 12-leg-joint diagnostic baseline against a 22-joint whole-body controller that allows arm motion. Gesture-conditioned walking, carrying objects, rough terrain, jumping, recovery, and hardware deployment require separate extensions; they are not accepted outcomes of this release. A controlled equal-budget arm ablation was proposed but is not claimed as completed.

## Model preparation and initial audit

The starting URDF has 28 links, 27 internal joints, and 28 STL meshes at scale 0.001. It declares approximately 11.714 kg, a zero-mass/zero-inertia root, and effort/velocity values of 100 for 26 bounded joints. These were not valid measured actuator limits. The builder subsequently reported approximately 3 kg assembled mass, a 160 g battery, ST3215 servos, and 7.2 V wall power. The simulation assumes the measurement included the battery and subtracts it once, giving 2.84 kg.

The final root budget is 0.70 kg; the remaining links share 2.14 kg. CAD geometry informed the root COM and inertia shape, but its uniform steel masses and duplicated components were rejected as measured mass. [Model_and_Hardware.md](Model_and_Hardware.md) records the assumptions and CAD comparison.

Use a genuinely floating base, with expected robot coordinates nq=34 and nv=33. The six free-base degrees of freedom are unactuated. Preserve geometry and explicit joint axes rather than guessing axis direction from names such as roll/pitch. The task frame is +X forward, +Y left, +Z up; the CAD frame is explicitly rotated into this convention. The original sole toe direction is local -Y.

Use simple checked collision proxies, including 105 x 180 mm sole geometry, rather than unrestricted mesh convex hulls. Enable ground and nonadjacent body collisions; exclude only structural adjacent overlaps. Find a bent-knee, level-sole reference with COM inside the support region, arm/knee clearance, and joint-limit margin. Validate CPU compilation, native gravity/contact behavior, independent resets, actuator ordering, and small GPU batches before PPO.

Position residual actions use `q_des = clip(q_ref + scale * a, soft_lower, soft_upper)`, with `a` clipped to [-1,1]. Leg, shoulder/elbow, wrist, head, and prismatic finger units and gains are handled separately. Reference-target low-pass filtering, rate limits, torque limits, and soft limits are retained. The 1.6 N m rotary cap and 4 rad/s target-rate cap are modeling assumptions, not measured servo ratings.

## Policy and observations

Start with 12 leg outputs for diagnosis, then use 22 outputs for 12 leg and 10 arm/wrist joints. The four finger slides and head hold reference targets while remaining in dynamics and joint-state observations. Never load a 12-output actor into a 22-output controller without an explicit validated mapping.

Initial observations were gyro, projected gravity, velocity command, all 27 relative joint positions and velocities, and the previous action, using five frames of history. Phase and heading were subsequently added. R10 added simulated body linear velocity to the actor, producing the final **460-dimensional actor**, **98-dimensional critic**, and **22-dimensional action** interface. Critic access to privileged simulation state is separate. Hardware requires a velocity estimator; the distance heatmaps are diagnostic displays rather than PPO inputs.

Training commands are sampled independently of the mouse. Learn idle and low-speed forward motion before expanding start/stop and bidirectional yaw. Zero commands are deliberately sampled. Initial broad targets up to 0.35 m/s and +/-0.6 rad/s were proposals; the published envelope is 0.12 m/s forward, -0.06 m/s reverse, and +/-0.25 rad/s yaw. Lateral translation is not part of the released joystick profile.

## Initial reward proposal

The v1 proposal used `r_step = dt_control * sum(w_i * f_i)` with `dt_control=0.02 s`. mjlab's reward manager already applies dt; functions must not multiply it again. Negative rewards remain negative. Empty swing sets return zero, avoiding NaN means. These weights document the starting proposal; the final revision-specific weights and functions are in `code/src/sunday_locomotion/env_cfg.py`, `mdp.py`, `quality_mdp.py`, and `control_mdp.py`.

| Term | Initial normalized definition | Initial weight | Gate or purpose |
| --- | --- | ---: | --- |
| Planar velocity | `exp(-||vxy_B-command_xy||^2 / 0.15^2)` | +2.0 | Track command in the same horizontal body frame |
| Yaw rate | `exp(-(wz-command_wz)^2 / 0.30^2)` | +1.0 | Track turns and suppress unwanted yaw |
| Upright torso | `exp(-(tilt/0.25)^2)` | +0.8 | Torso up axis relative to world vertical |
| Root height | `exp(-((h-h0)/(0.08*h0))^2)` | +0.4 | Allow normal gait variation |
| Reference posture | `exp(-mean(((q-qref)/sigma)^2))` | +0.2 | Moving sigma: legs 0.35, shoulder/elbow 0.30, wrist 0.15 rad |
| Alive | 1 | +0.1 | Smaller than tracking rewards |
| Quiet standing | `exp(-mean((qdot/qdot_scale)^2)-||vxy/0.05||^2)` | +0.3 | Zero command and both feet in contact |
| Action change | `mean((a-a_previous)^2)` | -0.05 | Dimensionless action smoothness |
| Torque use | `mean((actual_torque/limit)^2)` | -0.05 | Actual effort, not requested effort |
| Vertical velocity | `(vz/0.30)^2` | -0.10 | Suppress hopping |
| Torso roll/pitch rate | `||wxy/1.0||^2` | -0.05 | Do not penalize commanded yaw |
| Contact slip | `mean(contact * ||tangent_velocity/0.10||^2)` | -0.20 | Actual supporting feet, including at zero command |
| Swing height error | `mean(((hfoot-hclear)/0.02)^2)` | -0.10 | Moving swing phase; initial target 25 mm |
| Soft-limit overflow | `mean((overflow/(0.1*range))^2)` | -1.0 | Initial 90% soft range, plus physical hard bounds |
| Forbidden ground contact | Indicator of hand/head/torso/knee contact | -1.0 | Sustained contact also terminates |
| Forbidden self-collision | Nonadjacent checked collision pairs | -0.5 | Requires active collision configuration |
| Both feet airborne | Indicator of neither foot supporting | -0.3 | Filter solver contact noise |
| Optional contact phase | Match desired support phase | 0 initially; +0.2 if needed | Moving only; permit double support |
| Optional vertical angular momentum | `(Lz/(M*h0*0.30))^2` | 0 initially; -0.02 if needed | Weaken during intentional turns |
| Optional gait imitation | Model-specific joint/foot/contact error | 0 initially | Only after validating Sunday A1 reference feasibility |

Initial force-contact hysteresis proposed entry above 0.03 body weight, exit below 0.015 body weight, and two control steps of filtering. Contact forces are physical solver outputs; heatmap colors are not contact-force labels.

Later iterations introduced phase-conditioned foot trajectories, dense clearance deficits, landing braking, actual-load slip costs, quiet-stop and phase fade-out terms, yaw and velocity error costs, approximate mirror consistency, arm/leg and foot/foot clearance, and R25 continuous command-dependent cadence. Final cadence changes continuously from a 1.4 s period toward 1.0 s for larger commands. Reference motion shapes the reward and phase; it does not overwrite joint actions or drive the root.

## Termination and truncation

The initial design distinguished failure termination from time-limit truncation:

| Condition | Initial proposal | Treatment |
| --- | --- | --- |
| Episode time | Stand 10 s; walk 20 s | Truncate with correct time-out bootstrap |
| Large tilt | Over 60 deg for 0.10 s; over 80 deg immediately | Terminate and record reason |
| Low root | Below `0.55*h0` for 0.10 s | Terminate |
| Illegal ground support | Non-foot force over `0.02*M*g` for 0.04 s | Terminate |
| Nonadjacent self-collision | Sustained forbidden collision; v1 proposed a force threshold | Terminate or diagnose, according to the implemented revision |
| Nonfinite state/action/observation/reward | Any NaN or Inf | Fail and save diagnostics; do not hide it with replacement values |
| Implausible dynamics | Over 3 m/s root speed, 15 rad/s angular speed, or severe penetration | Diagnose numerical/model failure |
| Leaving local region | More than 5 m from environment origin | Truncate, distinct from a fall |
| Poor tracking alone | Large speed error or asymmetry | Initially penalize and measure, rather than reward frequent resets |

The current native viewer uses the actual thresholds in `native_web_player.py`: five control steps for tilt/height, two for illegal ground/self contact, immediate extreme tilt, and finite-state checks. Region/time boundaries are disabled for the continuously running demo, while fall/contact guards remain active. The training implementation is authoritative for its revision.

## Arms, grippers and head

The planned baseline holds arm targets while preserving their mass and collision shapes. The full-body variant learns ten arm/wrist outputs and permits asymmetric balance corrections. Do not assume named shoulder pitch axes correspond to intuitive human axes; check forward kinematics.

The final reference moves shoulder-pitch targets from 0 to 0.15 rad and elbow-pitch targets from 0.7 to 1.0 rad for clearance. CAD/encoder zero definitions do not change. The arms remain active policy outputs. Carrying objects and gesture modes require training with object mass, arm references and mode inputs; a fixed-base Week 3 grasp is not evidence of walking with a load. There is no independent waist joint in this model.

## Distance sensing

| Region | Preserved definition |
| --- | --- |
| Feet | Four 32 x 32 arrays, 2.5 mm spacing, original local sampling plane; `clip(max(-distance,0)/0.002,0,1)` |
| Hands | Four 12 x 32 arrays, 2 mm spacing; `clip(5000*max(0.0015-distance,0)/2.4,0,1)` |

Preserve Week 3 sampling geometry, transposition/orientation, fixed hand scale, and shared dynamic foot scale. Store raw responses separately from display color limits. A raised foot should clear its response; returned contact should restore the spatial pattern. Empty hands may correctly remain dark. No force coefficient is inferred before laboratory calibration.

Do not evaluate all 5,632 distance points in every parallel training environment. PPO uses lightweight physical contacts; complete tactile rendering is enabled for the selected playback environment. Final playback uses 500 Hz physics, 50 Hz policy control, and approximately 25 Hz scene/tactile updates from the same physical state.

## Curriculum and PPO budget

Proceed through model validation, standing and forward baseline, whole-body command learning, tactile playback, and interaction/acceptance. Increase command coverage only after measured quality checks. The initial PPO proposal used an ELU MLP (256,128,128), clip 0.2, gamma 0.99, GAE lambda 0.95, five epochs, four minibatches, learning rate 3e-4, adaptive KL target 0.01, entropy 0.01, and max gradient norm 1. Later revision-specific configuration is frozen with the selected policy.

Start with small GPU batches, measure memory/throughput on the 8 GB RTX 4070 Laptop GPU, and use bounded training segments. Final R25 used 512 environments and 32 control steps per rollout, planned 400 updates, and stopped after update 336. Checkpoint 300 was selected after evaluation; the highest numbered checkpoint was not automatically promoted.

The initial plan proposed gradual friction, mass/inertia, COM, gain, sensor-noise and delay randomization after basic walking, followed by multiple seeds for the selected setup. This release's 100-start tests randomize initial states; they are not a claim of completing every proposed domain-randomization or multi-training-seed study.

## Final interaction and acceptance

The original plan considered drag-speed/world-direction control. The implemented user requirement is a **hold-to-move, body-relative spring joystick**: displacement controls throttle/yaw, and release clears the request. Current body heading, head tracking, perspective correction, and camera independence are verified separately from locomotion.

Final acceptance requires command-sequence completion, randomized starts, quiet stopping, bidirectional stepping turns, reverse, independent native feedback, per-step clearance/contact quality, actual video inspection, and working heatmaps/UI. The versioned criteria and preserved original failure verdicts are documented in [Acceptance_Report.md](Acceptance_Report.md).
