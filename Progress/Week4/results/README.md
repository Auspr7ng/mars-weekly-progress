# Results and evidence

Start with the [acceptance report](../docs/Acceptance_Report.md) and [accepted policy manifest](../config/accepted_policy.json). Historical results are diagnostics; a survival count alone is not a walking-quality verdict.

## Final quantitative evidence

- [Twelve programs, 100 starts each](evaluations/control_r25_candidate300_responsive_controls_100.json): 1,200 completed/survived sequences.
- [Independent native CPU actor checks](evaluations/control_r25_candidate300_responsive_native_cpu.json).
- [Raw joystick evaluation](evaluations/control_r25_candidate300_responsive_joystick_100.json) and [whole-cycle assessment](evaluations/control_r25_candidate300_responsive_joystick_cycles_v2.json), with the [100-run NPZ](evaluations/control_r25_candidate300_responsive_joystick_100.npz).
- [Native 60-second joystick assessment](evaluations/r25_300_native_joystick_cycles_v2.json).
- [Slow gait audit](quality/control_r25_candidate300_responsive_steady_quality.json) and [fast gait audit](quality/control_r25_candidate300_responsive_fast_quality.json).
- [Original v1 result](control_r25_candidate300_responsive_result.json), which remains false, and the separate [final v2 assessment](control_r25_candidate300_responsive_result_assessment_v2.json).
- [Current joystick coordinate/browser result](validation/body_follow_v3_result.json), [four-heading regression](validation/body_follow_v3_contract.json), and [viewer timing](validation/final_web_realtime_single_client.json).

The original raw command reports, phase-aware reinterpretation and earlier failed results remain distinguishable. JSON filenames are retained for traceability. Large all-program per-step NPZ traces remain in the local archive, while the joystick trace and representative native/quality traces are included.

## Visual evidence

| Recording | Contents |
| --- | --- |
| [Speed/stop/restart](videos/control_r25_candidate300_responsive_responsive_speed_stop.mp4) | Speed changes and quiet stops |
| [Walking turns](videos/control_r25_candidate300_responsive_responsive_turns.mp4) | Both turn directions during forward walking |
| [In-place turns](videos/control_r25_candidate300_responsive_responsive_in_place.mp4) | Left and right stepping turns |
| [Reverse/stop](videos/control_r25_candidate300_responsive_responsive_backstep.mp4) | Reverse commands and stopping |

[Front/side review montage](videos/r25_300_all_controls_visual_review.png) and [current English joystick page](web_demo/body_follow_v3.jpg).

The videos document physical policy feedback. They are not joint animations or root-position playback. Heatmaps in these videos are the unchanged distance response.

## Tactile capture

[The saved capture](web_demo/capture_20261102_040715) contains four foot PNGs, two combined hand PNGs and raw arrays: hands `(4,384)`, feet `(4,1024)`, and 34 root/joint position coordinates. Saved heatmaps are uncalibrated distance responses.

## Training history

[run_metadata](run_metadata) includes compact settings and statuses for the historical trials. [training](training) contains the R25 event stream, [scalar CSV](training/r25_scalars.csv), and [training plot](training/r25_training_curves.png). Training reward is an optimization diagnostic and does not replace the acceptance suite.
