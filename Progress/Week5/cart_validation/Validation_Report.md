# cart_v2.xml validation report

November 12, 2026. Native MuJoCo 3.8.1, Python 3.11, CPU dynamics, timestep 0.002 s.

## Verdict and scope

The original XML fails to compile because the four caster swivel bodies have moving joints but no own geometry or inertial definition. The first error identifies `caster_fl` at source line 61. Child wheel masses do not provide the missing independent caster inertials.

The diagnostic copy remained numerically stable in the tested cases, but it has strong rolling resistance and slow caster reorientation. Whether the dynamics match the physical cart requires hardware measurements. This is independent model validation, not robot-cart pushing acceptance.

## Source

The model came from the lab's shared `Humanoid Locomanipulation/misc_3d_model/cart_v2.xml`. The [original](source/cart_v2.xml) remains unchanged, with SHA-256 `4fd8ffcf0ddc494a8c2d4eb0c744564d2a7a1fc7907fe91af512aa70e1f57370`. It has no external mesh/include dependencies and no ground plane.

```text
ValueError: Error: mass and inertia of moving bodies must be larger than mjMINVAL
Element name 'caster_fl', id 2, line 61
```

The same missing inertial structure occurs in `caster_fr`, `caster_rl`, and `caster_rr`.

## Structure and assumptions

| Item | Model setting |
| --- | --- |
| Platform | 220 × 310 × 10 mm, explicit platform mass 5 kg |
| Wheels | Radius 25.4 mm, width 25.4 mm, mass 0.2 kg each |
| Handle midpoint | Local position approximately (-0.210, 0, 0.195) m |
| Hand sites | `handle_left` and `handle_right`, Y offsets ±0.08 m |
| Joints | One freejoint, four swivel hinges, four roll hinges, 14 velocity DOFs |
| Drive | No actuators |
| Wheel axes | Cylinder orientation and local-Y roll hinge are parallel |
| Contact | condim=3, tangential sliding friction |

The test copy preserves original geometry, wheel locations, friction and damping. It adds each fork's assumed mass 0.05 kg, diagonal inertia (2e-5, 2e-5, 5e-6) kg·m², and inertial position (0.0075, 0, -0.0254) m. It also adds a floor with friction (0.8, 0.01, 0.001), lighting and rendering assets. The resulting total mass is 6.198829 kg. Handle masses are inferred from geometry/default density, so the total is greater than the platform's 5 kg label.

## Tests and measured results

Each force case settles for 2 s, receives the stated force at the handle midpoint during t=2–4 s, then releases through t=8 s. The resting case lasts 12 s. The coast case sets initial base velocity 0.2 m/s and consistent wheel speeds v/r at t=2 s, then evolves freely. No pose playback drives motion.

| Case | Observation |
| --- | --- |
| Rest, 12 s | About 1.4e-7 m horizontal drift, four settled wheel contacts, maximum tilt 0.030° |
| Initial 0.2 m/s coast | Only about 3.34 mm additional travel before stopping |
| 5 N forward | Mean forward speed 0.00844 m/s during t=3–4 s |
| 10 N forward | 0.02431 m/s, net forward travel 48.6 mm |
| 20 N forward | 0.06054 m/s, net forward travel 121.0 mm |
| 10 N reverse | -0.02432 m/s |
| 10 N sideways | Mean Y speed 0.00458 m/s, coupled translation/yaw |
| All casters initially 90°, 10 N forward | 0.00309 m/s, ending caster angles about 93.6–94.6° |
| Mixed caster headings, 10 N forward | 0.00785 m/s |
| Numerical checks | All 13 cases finite, zero warnings, no tipping, largest base-case tilt about 0.106° |

The 11 base cases also include diagonal force and 0.4 N·m yaw torque. Two extra straight-push cases use fork masses 0.01/0.10 kg each with proportionally scaled inertias. Their 10 N speeds are 0.02444/0.02417 m/s. This limited sensitivity test does not identify the real fork mass or inertia.

The [36-second video](results/cart_validation.mp4) contains rest, 10 N straight push, 10 N side push and transverse-caster push. The [summary](results/validation_summary.json) and individual case JSON files preserve force programs and sampled traces.

## Dynamics to review before next week

The original wheel-roll damping is 0.05 N·m·s/rad and friction loss 0.02 N·m per joint. Swivel damping/friction loss are 0.8/0.2. Under an ideal straight rolling approximation, the wheel damping corresponds to `4×0.05/0.0254² ≈ 310 N·s/m`, or approximately 31 N at 0.1 m/s. The four roll-joint dry friction terms add a nominal `4×0.02/0.0254 ≈ 3.15 N`. This is an explanatory approximation, not an exact force prediction.

Swivel resistance can oppose caster alignment. With zero swivel angle, the axle offset is +15 mm in local X. Confirm that the trail sign and initial wheel orientation match the hardware. The 90° test's limited rotation demonstrates slow reorientation without establishing which geometric change is appropriate.

condim=3 does not activate torsional/rolling contact-friction dimensions. Changing the second or third geom friction coefficient alone does not substitute for checking bearing-joint resistance. See [MuJoCo joint damping](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-damping) and [contact dimensions](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-condim).

Before robot integration, obtain measured caster fork mass/inertia, roll/swivel resistance, trail orientation and total payload. The commented hand weld template was not activated or tested. The original compile failure and physical parameter questions remain open.
