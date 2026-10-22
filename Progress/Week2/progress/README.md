# Week 2 progress — October 22, 2026

The [weekly slides](Weekly_Progress_2026-10-22_Four_FlexiTac.pptx) summarize four-finger virtual FlexiTac placement, the 12 × 32 taxel grids, and live open/close heatmaps. The corresponding implementation is in [Week 2 code](../code/README.md).

## Completed

- Configured separate sensors for the left/right upper and lower finger links.
- Displayed four live heatmaps and per-finger max, mean, and active-taxel counts.
- Recorded a gripper open/close demo. The open screenshot shows zero active taxels; the closed portion of the video shows all four panels responding.

## Evidence

- [Sensor placement](media/01_left_gripper_sensor_placement.png)
- [Taxel grid close-up](media/02_finger_taxel_grid_closeup.png)
- [Open-gripper heatmaps](media/03_four_heatmaps_open_gripper.png)
- [Open/close demo video](media/04_four_flexitac_open_close_demo.mp4)

The signals in this test measure normalized SDF proximity to the opposing finger, not calibrated pressure. The video is stored separately from the PPTX; the slides use a still from the recording.
