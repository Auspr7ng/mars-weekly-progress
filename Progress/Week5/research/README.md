# Foot tactile research — source review and proposed work

Read the [research plan](Foot_Tactile_Research_Plan.md). Its method remains a hypothesis that needs hardware data and controlled experiments.

Sources reviewed from the lab's shared folder:

| Source | What it establishes |
| --- | --- |
| `Code_yuvraj/32_16_fast.ino` | 32×32 ADC scan, AA55 marker, 1024 payload bytes, 2 Mbaud configuration |
| `Code_yuvraj/fast3.py` | Separate toe/heel streams, 30-frame median baseline, thresholding, spatial filtering and display smoothing |
| `Weekly_Yuvraj.pptx`, slides 3–6 | Historical demonstrations and a reported duplicate toe response requiring a fresh check |
| `Gelsight/left_foot_solid.step`, `sensor assembly.step` | Mechanical references, not a force calibration |
| `TacVLA/TacVLA-main.zip` | A manipulation example with a 15×8 tactile encoder and a contact gate based on at least three taxels over 0.1 |

Full original files and their hashes remain in the local Week5 research archive. Other people's presentations, private lab links and the full reference archive are not redistributed here. The source inventory describes reviewed material; it is not proof of the current hardware's performance or installation.

The local frozen simulation code is available publicly in [Week4](../../Week4/code/src): `env_cfg.py`, `mdp.py`, `native_observations.py`, and `week3_tactile/{feet_tactile,flexitac}.py`.
