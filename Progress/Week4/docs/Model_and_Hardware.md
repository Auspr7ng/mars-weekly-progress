# Model, CAD and hardware assumptions

## Accepted simulation mass budget

The robot builder reported approximately **3 kg** assembled mass, a **160 g** battery, **ST3215** servos, and **7.2 V** operation. The URDF effort/velocity values of 100 were confirmed placeholders. The simulation assumes the weighing included the battery and subtracts it once for wall-powered operation, giving **2.84 kg**. This assumption still needs a consistent hardware weighing convention.

The later metadata assigns approximately **0.70 kg** to the central root link. The final model allocates **2.14 kg** to the remaining links. Their previous relative mass/inertia distribution was scaled together by approximately **0.8290759378**. The root COM estimate in the task frame is `[-0.002407317, -0.000127605, 0.084750202] m`. Root inertia uses CAD shape scaled to the root mass and rotated into the task frame. These are uniform-material geometric estimates, not measured link inertias.

Physical MJCF SHA-256: `c1168aee2b339adc15fa8fc3338e42c729dc92f0db07891c685cf96ca604bcd2`. The zero-mass CAD wrapper beneath the massive free root is a coordinate wrapper; the simulated floating root itself has nonzero mass and inertia.

## Native CAD audit

The complete assembly was inspected through the native Fusion API. All 27 internal joints were read: 23 rotary joints including the head and four finger slides. Twenty-six joints had enabled bounds; the head did not. Enabled bounds matched the source URDF within 4.91e-7, and the then-current MJCF within 4.50e-6, consistent with rounding.

| Joint suffix | Right range | Left range | Units |
| --- | --- | --- | --- |
| shoulder_roll | -85 to 85 | -85 to 85 | deg |
| shoulder_pitch | -15 to 135 | -15 to 135 | deg |
| elbow_yaw | -90 to 90 | -90 to 90 | deg |
| elbow_pitch | -90 to 75 | -90 to 75 | deg |
| wrist_yaw | -90 to 90 | -90 to 90 | deg |
| hip_roll | -80 to 80 | -80 to 80 | deg |
| hip_pitch | -130 to 45 | -45 to 130 | deg |
| thigh_yaw | -45 to 45 | -45 to 45 | deg |
| knee_roll | -60 to 90 | -90 to 60 | deg |
| ankle_roll | -30 to 90 | -30 to 90 | deg |
| ankle_pitch | -30 to 30 | -30 to 30 | deg |
| finger_slide_top/bottom | 0 to 30 | 0 to 30 | mm |
| head_rotation | Bounds disabled | — | — |

The simulation head range of +/-1.2 rad is an explicit modeling bound; it was not verified as a hardware/CAD limit. Disabled CAD bounds storing 0/0 do not mean the head is locked.

Parent/child ordering and unit joint axes matched the URDF; maximum axis-vector difference was 5.84e-7. CAD axis tables use assembly-world coordinates at the saved zero, rather than each link's local frame. The first hip axes are oblique: about `[-0.766044,0,0.642788]` right and `[0.766044,0,0.642788]` left. Joint names do not justify replacing them with pure X/Y axes.

All saved rotary/slide values were zero. Compared with URDF zero configuration, rotary pivot positions differed by at most 0.54 micrometres and child-link geometric centres by at most 0.36 micrometres. Slide reference origins may include fixed offsets; axes, travel and actual geometry were compared rather than treating an origin offset as a zero error. Only left elbow yaw had an enabled Rest value, equal to zero.

CAD zero, encoder zero and the bent-knee training reference are different concepts. Encoder offsets and physical positive rotation direction still require hardware calibration. Four finger slides had no CAD MotionLink coupling; the rotation-to-translation mapping of each hand's single motor remains uncalibrated.

## Why CAD mass was rejected

| Native CAD quantity | Value | Interpretation |
| --- | ---: | --- |
| Complete assembly mass | 32.965585 kg | Includes duplicate geometry |
| One joint-connected set of 28 components | 16.482773 kg | Still evaluated as steel |
| Additional geometry | 28 components | Duplicate set, plus three empty components |
| Active root mass | 4.768683 kg | Ten root solids assigned steel |
| Active solid materials | 57 steel solids, 7,850 kg/m3 | Inconsistent with actual material/electronics distribution |

None of these steel/duplicate values replaces the approximate measured 3 kg or root metadata of 700 g. The old 11.714 kg URDF declaration is close to the single CAD set with its root omitted and is not a weighing result. Native inertia values were converted from kg cm2 to kg m2 by 1e-4, shifted to COM with the parallel-axis theorem, and checked against principal inertias.

## Actuators and mapping

The simulation uses recorded conservative assumptions: rotary torque caps around 1.6 N m, a 4 rad/s reference-target rate cap, group-specific PD gains, target low-pass factor 0.3, and 90% soft position limits. These are not measurements of servo torque/speed capability. Fingers use separate linear units. The final policy has 22 active leg/arm/wrist outputs; head and finger targets are held separately.

The metadata text says ST3125 while its link points to ST3215; the builder previously confirmed ST3215. The document also says 7.4 V, versus the earlier 7.2 V confirmation. These discrepancies remain recorded; no new measured motor curve was inferred from them.

Hardware identifiers are board plus channel. The two arms use boards 1/2, channels 1–6; legs use board 3, right 7–12 and left 19–24. Mapping follows the chain from torso toward extremity rather than similar names. For example, hardware shoulder-pitch maps to the URDF's first shoulder-roll joint. Each hand's channel 6 corresponds to two model slides; the head channel was not provided. [Structured candidate mapping](../config/hardware_joint_mapping_draft.json).

Derived tables: [joint limits/axes/zeros](../research/hardware_metadata/2026-11-01/fusion_joint_limits_axes_zero.csv), [link mass properties](../research/hardware_metadata/2026-11-01/fusion_link_mass_properties.csv), and [comparison data](../research/hardware_metadata/2026-11-01/fusion_verified_comparison.json). Original CAD and Word files remain local; this release contains their derived technical audit.

## Remaining hardware work

Confirm the battery/root weighing convention, per-link mass distribution, root COM/inertia, exact servo model and operating voltage, encoder directions/offsets, finger coupling, and a base-velocity estimator. Calibrate tactile force-response coefficients in the laboratory. The current acceptance is for flat-ground simulation only.
