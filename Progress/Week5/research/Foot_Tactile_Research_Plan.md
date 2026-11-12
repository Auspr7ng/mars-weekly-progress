# Foot tactile sensing: proposed method and experiment plan

November 12, 2026. **Status: source review and method design.** No new hardware dataset, validated foot detector, force calibration or tactile policy result is claimed. The original questions came from the project author and remain research questions.

## Research hypothesis

A stable foot-contact feature shared by hardware and simulation may be easier to transfer than the current geometric distance heatmap. First test contact detection. Then test whether that information improves walking. Toe/heel features and high-resolution pressure reconstruction remain conditional extensions.

Define the label before collecting data. Geometric touch and load-bearing support are different. A sensor with weak sensitivity to light contact can read zero while the foot is touching. The first proposed target is effective support contact, with light touch and near-ground states labelled separately. Agree on the reference load and label definition before evaluating detectors.

## Source facts

- The reviewed firmware configures 32 rows and 32 columns despite its `32_16_fast.ino` filename. It shifts 10-bit ADC values to 8-bit and sends AA55 plus 1024 bytes. Baud rate alone does not establish sample rate or latency.
- `fast3.py` handles toe and heel on separate serial ports. It takes an initial 30-frame median, subtracts a default 15-count threshold, clips positive responses to 100 and scales to 0–1. It removes isolated cells. Display smoothing uses alpha 0.2, and the toe image is vertically flipped.
- The variable name `pressure` does not establish force or pressure calibration. The two reviewed acquisition/display files do not provide a validated per-foot contact state machine or stale-frame validity output.
- A historical sensor presentation reports duplicate toe responses and suggests a board issue. The cause and present status require a new location-controlled test.
- The reviewed TacVLA example gates manipulation tactile tokens when at least three taxels exceed 0.1. Its 15×8 array, task and raw value range differ from Sunday A1's feet. These thresholds are a structural reference, not transferable calibration values.
- Mechanical CAD filenames do not establish that the ADC array and a GelSight optical sensor are the same hardware stack. Pad dimensions, material, mounting preload and load path remain unconfirmed.

## Current simulation audit

Week5 reuses the frozen Week4 R25-300 policy. `env_cfg.py` and `native_observations.py` show that the actor receives no tactile heatmap, foot-contact bit or sole force. The critic's `foot_state` includes foot height, contact and normalized force. Contact forces also contribute to reward terms.

`mdp.py` declares contact when the magnitude of the mean foot contact-force vector exceeds 0.03 mg. Under the existing 2.84 kg model budget, that is about 0.84 N. It is a simulation threshold, not an ADC threshold or a hardware calibration.

The four sole patches each use 32×32 samples at 2.5 mm pitch with an 80 mm nominal patch size. `feet_tactile.py` uses a zero shell and 2 mm response depth. `flexitac.py` maps signed geometric distance to `clip(max(-d,0)/0.002,0,1)` for the sole response. It does not model the electrical sensor response or a force per taxel. The viewer shares an automatic foot colour range, so raw arrays and fixed scales are needed for quantitative comparisons across time.

Thus geometry drives visualization, while the contact solver supplies separate physical signals. Similar grid dimensions or colours do not establish equivalent measurements. The actor's simulated base-velocity feedback also needs a hardware estimator before physical deployment.

## Candidate foot-contact detectors

| Candidate | Proposed computation | Main question |
| --- | --- | --- |
| D1: active area | Correct unloaded baseline and response direction, apply taxel noise thresholds, then measure effective area or connected regions with temporal hysteresis | Can it reject noise without missing small real contacts? Can duplicate responses inflate area? |
| D2: aggregate response | Sum or robustly aggregate valid positive responses with separate entry/exit thresholds and persistence times | Does it remain repeatable under load position, drift, saturation and gain differences? |

For either method, first determine whether loading increases or decreases each channel. A schematic positive residual is `r_i=max(s_i*(raw_i-baseline_i),0)`, where the sign comes from measurements. Keep toe/heel intermediate features, then combine them into a per-foot state. Fit thresholds on training/calibration sessions and evaluate on separate collection sessions, not a random split of neighbouring frames.

The initial output is contact or a response score. Newtons or pascals require independent calibration and validation. A response-weighted centroid is not automatically the physical centre of pressure.

## Phase 1: hardware contract and labelled data

Confirm whether both feet have all four pads. The reviewed two-port example demonstrates one toe/heel pair, not necessarily a complete four-pad deployment. Record pad dimensions, orientation, protective layers, preload and any rigid structure that bypasses the sensor.

Collect unloaded noise/drift, repeated local loading at the centre/corners/edges, several repeatable load levels, and loading/unloading holds. Reproduce the toe duplicate-response report. Capture supported foot lift, light touch, loading, unloading, toe-only, heel-only and full-foot support with synchronized reference force or scale readings and time-aligned video. Do not label truth using the same heatmap threshold being tested.

Preserve timestamps, frame IDs, sensor/foot/region IDs, raw arrays, baseline version, labels/reference loads, validity and dropped-frame counts. Establish baselines under confirmed unloaded conditions so standing load is not subtracted away. Measure actual acquisition intervals and end-to-end delay.

The current viewer smooths `shown` during drawing, while `pressure` contains a different processing stage. A control interface needs an explicit signal stage and filter time constant based on timestamps, rather than GUI frame rate. The existing marker/length protocol lacks a frame counter/checksum in the reviewed firmware. Measure framing errors before deciding whether to extend it.

## Phase 2: offline detector comparison

Compare D1 and D2 on the same labelled data. Evaluate false contact in air, missed loaded support, precision/recall, light-touch confusion, contact-on/off delay and state chatter. Test drift and loading-position changes on held-out sessions. Agree on acceptance criteria before examining test results. Set latency requirements using measured sensor timing and the control cycle rather than promising an unmeasured 20 ms response.

Export contact together with validity and age. A disconnected or stale sensor is unknown, not automatically an airborne foot. Define the control system's response to invalid input separately.

## Phase 3: matched simulation observations

Use the same feature meaning and processing contract in both domains. Build the simulation feature from physical contacts, then model the measured sensor's sampling, threshold, hysteresis, noise, delay and outages. Perfect simulator contact bits alone do not reproduce the hardware detector.

Candidate implementations are contact-normal-force aggregation at each sole region or MuJoCo touch sensors with carefully checked site coverage. A touch sensor sums included normal contact forces. Verify support in the installed native/mjlab/Warp path before choosing an implementation. [MuJoCo touch documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch).

Do not split a whole-foot force into arbitrary toe/heel fractions. Region estimates require contact positions or independently observable region signals. Neither sparse contact forces nor joint geometry uniquely determine a continuous 32×32 pressure distribution. [MuJoCo contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#soft-contact-model).

Retain the existing distance maps for geometry/orientation inspection. A later pressure-map project would need a calibrated spatial response or compliant material model, reference loads, total-load consistency and checks for crosstalk, hysteresis and saturation. Multiplying the current distance response by a constant would not establish pressure calibration.

## Phase 4: policy comparisons

| Experiment | Actor information | Question |
| --- | --- | --- |
| P0 | Existing actor, no tactile | What is the unchanged baseline? |
| P1 | Per-foot contact features and validity, with measured observation imperfections | Does contact information improve support transitions, landing or stopping? |
| P2, conditional | Toe/heel contact features and calibrated regional load if available | Does regional information add value beyond P1? |

Keep command programs, dynamics, rewards, seeds and training/evaluation budgets comparable. Measure falls, slip, contact-phase errors, tracking and stop stability on independent seeds and perturbations. Do not assume tactile must improve results.

The critic may retain privileged simulator signals. The actor must use deployable observations. Adding tactile changes input dimension, history, normalization and weights and therefore needs training or fine-tuning. The current 460-dimensional policy cannot simply accept additional hardware fields. If P1 adds no value, investigate information quality, redundancy and delay before escalating to a high-dimensional encoder.

## Work plan and decision gates

| Stage | Deliverable | Gate |
| --- | --- | --- |
| 1. Interface audit | Hardware/simulation field table and capture format | Region, units, orientation and available pads are explicit |
| 2. Bench collection | Labelled raw data, noise/drift/timing measurements | Unloaded/light-touch/support states have independent labels |
| 3. Detector comparison | D1/D2 report and parameter evidence | Held-out error and latency meet pre-agreed criteria |
| 4. Simulation observation model | Feature comparisons and region tests | Shared semantics and measured observation limitations |
| 5. Policy and supported hardware check | P0/P1 results, P2 if justified, deployment-field logs | New policy/input chain passes before free walking |

These are dependent stages, not a claim that all can finish this week. The priority is the interface table, labelled data and detector comparison. If hardware data is unavailable, deliver the capture/annotation and evaluation plan without calling it hardware validation. No detector or policy change was implemented during this review.
