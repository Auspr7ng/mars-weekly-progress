# Training history and decisions

The table consolidates the local R1–R25 notes. Intermediate checkpoints and their large trajectories remain in the local archive. Machine-readable run configurations, stop states and initialization records are included under [results/run_metadata](../results/run_metadata). Different rounds changed multiple conditions, so these observations do not establish single-factor causal effects.

| Round | Main change | Recorded outcome or reason to continue |
| --- | --- | --- |
| R1 | Floating-base standing and forward experiments; measured 2.84 kg mass interpretation replaced the original CAD mass | Standing could be rewarded without useful walking; pre-measurement checkpoints were not resumed as measured-model results |
| R2 | Bent-knee stance, phase and stronger movement incentives | Standing at a 0.1 m/s command still earned 0.641 from the initial broad velocity kernel. Later motion included sliding/hopping; checkpoint 100 moved about 4.07 m in 20 s with 0.221 m/s speed error. Stopped at 265 |
| R3 | Target low-pass factor 0.3, denser speed error, contact slip and support-quality constraints | Checkpoint 599 survived 100/100 and tracked forward speed, but mean maximum lateral drift was 0.324 m. More iterations did not fix heading |
| R4 | Added heading sin/cos observations and heading/lateral penalties | Checkpoint 399 met the then-used numerical screen: 100/100 survival, 1.663 m forward, 0.01496 m/s speed RMSE, 0.0769 m mean maximum drift. Visual gait acceptance was later withdrawn |
| R5 | Explicit foot trajectories, 1.4 s cadence, stance/sway and landing quality | Native and GPU audits exposed excessive roll, slip, impact and asymmetric support in R4. A temporary 0.98 soft-limit factor was explored; it was not retained |
| R6 | Corrected the knee-flexion branch and restored 0.90 soft limits | The earlier branch was an implementation error. Correct geometry could clear the full sole about 20 mm, but the learned policy still lifted too little. Stopped at 519 |
| R7 | 22 active leg/arm/wrist outputs, bent elbows, exploration and support-gated tracking | Checkpoint 200 had roughly 27–29 mm swing peaks but 12.5 deg roll and poor touchdown braking. Stopped at 247 |
| R8 | Actual-load slip cost, late-swing braking, torso damping and corrected acceleration normalization | Landing improved; checkpoint 100 still failed lateral drift. Checkpoint 200 seeded later work. Stopped at 272 |
| R9 | Reduced sway from 28 to 18 mm and penalized drift beyond planned sway | Checkpoint 100 retained about 0.55 m maximum drift. Stopped at 247 |
| R10 | Added simulated body linear velocity to the actor, expanding history from 445 to 460 dimensions | Mapping was numerically verified, but checkpoint 100 still had about 0.441 m mean maximum drift. Native feedback was corrected to XBODY rather than the inertial COM frame. Stopped at 216 |
| R11 | Tracked the derivative of the intended lateral load-transfer motion | Checkpoint 100 survived 100/100, moved 1.184 m in 20 s, and reached the fixed-speed gait screen. It was a 0.06 m/s baseline, not a stop/turn policy. Stopped at 120 |
| R12 | Combined speed, stop and turning commands | 600 updates completed, 9,830,400 transitions. Four programs each survived 100/100, but only steady walking fully passed; stopping retained periodic shuffling |
| R13 | Quiet-stop rewards, foot stillness and phase fade-out | Stop stages passed 100/100, but overall speed, drift, bilateral stepping and clearance remained inadequate. Five programs survived; only idle fully passed |
| R14 | More bidirectional turn samples, stronger clearance and lateral-speed costs | Checkpoint 150 reduced steady drift to about 0.041 m in a small screen, but yaw variation and turn stepping still failed. Stopped at 178 |
| R15 | Larger turning load transfer, denser sway/yaw/clearance costs | Left stepping improved, while in-place forward drift and slow right turns persisted. Static IK was treated as diagnostic, not proof of dynamic feasibility. Stopped at 197 |
| R16 | Removed a clearance-cost plateau, used continuous phase scores and in-place/stop-transition costs | Checkpoint 175 improved many local checks, but drift and right-turn stepping still failed |
| R17 | Added approximate mirror-consistency loss with CAD-axis signs and reference offsets | Straight and speed/stop screens improved. Geometry was only approximately symmetric; reflected trajectories were not treated as real PPO samples. In-place turns still lacked bilateral stepping |
| R18 | Dense unloading feedback and small phase-dependent turn-lean reward targets | Training paused at update 52 when new metadata/CAD data arrived. This was a requested pause, not completed acceptance |
| R19 | Applied a 0.70 kg root budget and CAD-shape COM/inertia estimate within the 2.84 kg total | Revalidated native/GPU standing and retrained on the changed dynamics. Old-dynamics passing results did not transfer automatically |
| R20 | Dense hand/leg clearance | Contact/clearance problems motivated further training; no promotion based only on reward |
| R21 | Included nominal reference starts alongside randomized starts | R21-275 became the earlier accepted seven-program profile: 700/700. Its slower profile was protected during subsequent trials |
| R22 | Hold/release joystick curriculum, stronger command slew and a wider experimental speed/yaw range | Moderate control improved; the full 0.15 m/s, +/-0.4 rad/s envelope still had collision/quality failures. It seeded R25 at checkpoint 275 |
| R23 | Expanded action authority for larger commands; endpoint-balanced sampling and clearance costs | Hand/knee and foot/foot failures showed that larger residual ranges were not a sufficient fix |
| R24 | Safer shoulder/elbow reference and gradual ankle authority | Four reference changes improved arm clearance; fast foot placement/turn quality still failed |
| R25 | Restored original action ranges; introduced continuous command-dependent cadence | Selected R25-300 passed the final 12-program, native/CPU, cycle-aware joystick, gait and visual checks at the published narrower envelope. Training stopped after 336 |

## Why numerical survival was insufficient

The R4 visual verdict was withdrawn after additional scrutiny. Its GPU audit had approximately 14.03 deg roll variation, 8.49 deg pitch variation, 0.0541 m/s supporting-sole speed p95, and impact near 2.06 body weights. Independent native physics showed similar defects. Surviving and moving forward did not establish acceptable stepping.

| Diagnostic model | Command | Duration | Minimum swing peak L/R | Roll range | All-contact support speed p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R6-500 | 0.08 m/s | 20 s | 5.6 / 7.8 mm | 8.0 deg | 0.0587 m/s |
| R7-200 | 0.06 m/s | 12 s | 28.8 / 27.0 mm | 12.5 deg | 0.0491 m/s |
| R8-100 | 0.06 m/s | 20 s | 23.0 / 23.7 mm | 9.2 deg | 0.0106 m/s |
| R8-200 | 0.06 m/s | 12 s | 19.1 / 22.8 mm | 7.4 deg | 0.0375 m/s |

These historical all-contact metrics should not be silently relabeled as the later settled-stance metric. The final criteria distinguish touchdown from planted support and use whole-cycle speed means. The original v1 final verdict remains false in the archived raw JSON; the separate v2 verdict records the methodology change.

## Current interaction

The final locomotion checkpoint and the viewer are versioned separately. `body_follow_v3` corrects frozen drag-frame/anchor behavior and off-centre perspective mapping. The base and axes follow current head/body state while held. Four heading poses and actual browser forward/steering/off-centre/release checks are recorded in [the interaction result](../results/validation/body_follow_v3_result.json).

See the English [R22](Round22_Deadman_Fast_Joystick.md), [R23](Round23_Action_Range.md), [R24](Round24_Arm_Reference_Clearance.md), and [R25](Round25_Command_Cadence.md) design notes for the later experiments. Their proposed wider envelopes are historical training targets, not the released limits.
