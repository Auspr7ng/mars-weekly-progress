# MARS weekly progress

Weekly research notes, code, slides, and visual evidence for the Sunday A1 / FlexiTac work in Isaac Sim and mjlab.

Weeks 1 and 2 depend on local robot assets and an Isaac Sim environment. Week 3 includes the Sunday A1 model resources and a portable mjlab demo. Use an existing compatible mjlab environment and an NVIDIA GPU, then open `Progress/Week3/start_demo.bat` on Windows. See the [Week 3 README](Progress/Week3/README.md) for requirements and Linux commands.

Week 4 adds whole-body RL walking, speed/stop/reverse/turn control, a body-relative spring joystick, and the original eight distance heatmaps. Its English release includes the accepted checkpoint, frozen framework/viewer sources, model assets, evaluation reports and control recordings. Start at the [Week 4 guide](Progress/Week4/README.md).

Week 5 adds Bluetooth Xbox control through the same walking policy, a foot-tactile research plan, and independent cart model validation. The release includes English code documentation, recorded evidence, and an eight-slide weekly presentation. Start at the [Week 5 guide](Progress/Week5/README.md).

## Weeks

| Week | Code | Progress |
| --- | --- | --- |
| Week 1 | [Two-foot sensor setup](Progress/Week1/code/README.md) | [Slides and evidence](Progress/Week1/progress/README.md) |
| Week 2 | [Four-finger sensor setup](Progress/Week2/code/README.md) | [Slides, screenshots, and demo](Progress/Week2/progress/README.md) |
| Week 3 | [mjlab dual-grasp and eight-array demo](Progress/Week3/README.md) | [Weekly slides](Progress/Week3/Weekly_Progress_2026-10-29_mjlab_Tactile.pptx) and [migration report](Progress/Week3/Week3_Migration_Report.md) |
| Week 4 | [RL walking and robot-relative joystick](Progress/Week4/README.md) | [Acceptance report](Progress/Week4/docs/Acceptance_Report.md), [training history](Progress/Week4/docs/Training_History.md), and [results](Progress/Week4/results/README.md) |
| Week 5 | [Xbox control, tactile research, and cart validation](Progress/Week5/README.md) | [Weekly slides](Progress/Week5/Weekly_Progress_2026-11-12_Xbox_Tactile_Cart.pptx), [tactile plan](Progress/Week5/research/Foot_Tactile_Research_Plan.md), and [cart report](Progress/Week5/cart_validation/Validation_Report.md) |

Weeks 1 and 2 separate runnable scripts in `code/` from reports in `progress/`. Week 3 keeps `start_demo.bat`, the README, report, slides, and selected media together, with implementation in `code/` and saved evidence in `results/`.

## Add a new week

1. Create `Progress/WeekN/code/` and `Progress/WeekN/progress/media/`.
2. Copy the two [README templates](templates/) into the corresponding folders and fill in the results and limitations.
3. Put runnable scripts in `code/`, the weekly PPTX in `progress/`, and screenshots or short recordings in `progress/media/`.
4. Review files before committing. Keep credentials, personal data, caches, and generated simulation outputs out of Git. For large videos or USD assets, use Git LFS or a separate storage link.
5. Run `git add Progress/WeekN README.md`, `git commit -m "Add Week N progress"`, and `git push`.

## Data interpretation

The Week 1 two-foot setup uses a provisional aggregate-force calibration. The Week 2 finger heatmaps and Week 3 hand/foot heatmaps are normalized geometry-based responses, not calibrated pressure or force measurements. Week 3 obtains physical grasp forces separately from the MuJoCo contact solver. See each week's README for details.

Week 4 preserves the Week 3 distance response and reports physical contact metrics separately. Its acceptance is limited to flat-ground simulation; it does not establish calibrated tactile pressure or hardware walking.

Week 5's tactile method remains a proposal awaiting labelled hardware data. Cart results use a diagnostic copy with assumed caster inertials because the original XML fails to compile. Robot-cart pushing remains next week's work.
