# MARS weekly progress

Weekly research notes, code, slides, and visual evidence for the Sunday A1 / FlexiTac work in Isaac Sim.

This is a private working archive. The scripts depend on local robot USD/URDF assets and an Isaac Sim environment; cloning this repository alone does not reproduce the simulation.

## Weeks

| Week | Code | Progress |
| --- | --- | --- |
| Week 1 | [Two-foot sensor setup](Progress/Week1/code/README.md) | [Slides and evidence](Progress/Week1/progress/README.md) |
| Week 2 | [Four-finger sensor setup](Progress/Week2/code/README.md) | [Slides, screenshots, and demo](Progress/Week2/progress/README.md) |

Each week has two separate READMEs: `code/README.md` explains how to run and interpret that week's script, while `progress/README.md` records what was completed and links to the presentation and media.

## Add a new week

1. Create `Progress/WeekN/code/` and `Progress/WeekN/progress/media/`.
2. Copy the two [README templates](templates/) into the corresponding folders and fill in the results and limitations.
3. Put runnable scripts in `code/`, the weekly PPTX in `progress/`, and screenshots or short recordings in `progress/media/`.
4. Review files before committing. Keep credentials, personal data, caches, and generated simulation outputs out of Git. For large videos or USD assets, use Git LFS or a separate storage link.
5. Run `git add Progress/WeekN README.md`, `git commit -m "Add Week N progress"`, and `git push`.

## Data interpretation

The Week 1 two-foot setup uses a provisional aggregate-force calibration. The Week 2 finger heatmaps are normalized SDF proximity/contact proxies, not calibrated pressure or force measurements. See each code README for details.
