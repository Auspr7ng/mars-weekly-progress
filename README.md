# MARS weekly progress

Weekly research notes, code, slides, and visual evidence for the Sunday A1 / FlexiTac work in Isaac Sim and mjlab.

Weeks 1 and 2 depend on local robot assets and an Isaac Sim environment. Week 3 includes the Sunday A1 model resources and a portable mjlab demo. Use an existing compatible mjlab environment and an NVIDIA GPU, then open `Progress/Week3/start_demo.bat` on Windows. See the [Week 3 README](Progress/Week3/README.md) for requirements and Linux commands.

## Weeks

| Week | Code | Progress |
| --- | --- | --- |
| Week 1 | [Two-foot sensor setup](Progress/Week1/code/README.md) | [Slides and evidence](Progress/Week1/progress/README.md) |
| Week 2 | [Four-finger sensor setup](Progress/Week2/code/README.md) | [Slides, screenshots, and demo](Progress/Week2/progress/README.md) |
| Week 3 | [mjlab dual-grasp and eight-array demo](Progress/Week3/README.md) | [Weekly slides](Progress/Week3/Weekly_Progress_2026-10-29_mjlab_Tactile.pptx) and [migration report](Progress/Week3/Week3_Migration_Report.md) |

Weeks 1 and 2 separate runnable scripts in `code/` from reports in `progress/`. Week 3 keeps `start_demo.bat`, the README, report, slides, and selected media together, with implementation in `code/` and saved evidence in `results/`.

## Add a new week

1. Create `Progress/WeekN/code/` and `Progress/WeekN/progress/media/`.
2. Copy the two [README templates](templates/) into the corresponding folders and fill in the results and limitations.
3. Put runnable scripts in `code/`, the weekly PPTX in `progress/`, and screenshots or short recordings in `progress/media/`.
4. Review files before committing. Keep credentials, personal data, caches, and generated simulation outputs out of Git. For large videos or USD assets, use Git LFS or a separate storage link.
5. Run `git add Progress/WeekN README.md`, `git commit -m "Add Week N progress"`, and `git push`.

## Data interpretation

The Week 1 two-foot setup uses a provisional aggregate-force calibration. The Week 2 finger heatmaps and Week 3 hand/foot heatmaps are normalized geometry-based responses, not calibrated pressure or force measurements. Week 3 obtains physical grasp forces separately from the MuJoCo contact solver. See each week's README for details.
