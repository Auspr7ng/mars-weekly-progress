# Week 6 — Jetson R25 bench access and Xbox input

**Progress date: November 18, 2026.** The Jetson Orin Nano Developer Kit is
reachable from the Mac over USB-C at `192.168.55.1`. The existing R25-300
deployment package was inspected on the Jetson, and two no-motor shadow runs
completed. A separate Xbox input script is ready for controller identification
and shadow testing when the controller is powered and connected to Jetson.

| Area | Current result |
| --- | --- |
| Jetson access | USB device, serial, USB Ethernet and SSH verified; Ubuntu 22.04.5 LTS on aarch64 |
| R25 package | Existing verified-file checks passed; actor input/output is 460/22 and checkpoint SHA-256 matches the Week 4 accepted model |
| Sensor and policy diagnostic | Two 2-second, 100-step shadow runs completed with `motor_output=False` |
| Xbox control | [Standalone Jetson script and instructions](code/README.md) written; five focused input tests pass locally and on Jetson |
| Motor preview | Suspended Home preview was interrupted: right elbow stayed 619 ticks from Home, and torque-off acknowledgement failed for motor ID 4; motor power cutoff was requested |
| Physical walking | No ground-walking acceptance or policy motor run was performed in this session |

[Detailed progress and next checks](progress/README.md).

[R25 policy sensor and servo data requirements](policy_inputs/README.md).

## Open Jetson code in Mac VS Code

The Mac has VS Code Remote SSH installed. With the USB connection active, run
this on the Mac to open the Jetson's deployment folder directly:

```bash
code --new-window --remote ssh-remote+sunday@192.168.55.1 /home/sunday/Sunday_v13_Jetson_Rollout
```

Enter the SSH password in VS Code when prompted. The window should show
`SSH: 192.168.55.1`; its Explorer and terminal then operate on Jetson files.
Running plain `code .` in a Mac terminal opens a Mac folder instead.
