# Week 3 migration report

## Outcome and scope

The latest complete Sunday A1 model now runs in a fixed-base mjlab scene with independent left and right grippers, two free spheres, four finger tactile arrays, and four sole tactile arrays. One Viser interface provides gripper controls, independent ball resets, ball offsets, and eight tactile arrays. The spheres are 28 mm in diameter and 40 g each. Their grasping, release, and falling are computed by the physical solver.

The task uses 27 position actions. A physical time step is 2 ms; ten physical steps form one 20 ms control step. Tactile observations update with the control steps and the interface refreshes at approximately 10 Hz. A CPU MuJoCo data copy is used to display the GPU state and is not advanced as an independent simulation.

## What changed for the hands and feet

Our Week 2 hand setup used four independent finger links, with one 12 x 32 grid per finger and 2 mm spacing. That layout is retained: four arrays contain 1,536 sampling points. A cell-based coverage convention gives a 24 x 64 mm patch; the span between the first and last sampling centers is one spacing smaller.

Our Week 1 foot setup used one 12 x 32 grid per foot at 5 mm spacing, querying the ground mesh with a signed-distance method. The new layout has separate front and heel patches on each foot. Each patch has 32 x 32 points at 2.5 mm spacing, covering 80 x 80 mm by the same cell-based convention. The two patches on each foot have a 16 mm center gap between their edges and approximately 2 mm clearance from the toe and heel ends. The four foot arrays contain 4,096 sampling points.

Foot patches are attached to `l_foot_link_1` and `r_foot_link_1`. Their local plane is Z = -30 mm. Front and heel center Y coordinates are -99.175 mm and -3.175 mm. Left and right center X coordinates are +5 mm and -5 mm. The toe direction is local -Y. Grid points follow the current rigid-body poses instead of remaining fixed in world coordinates.

## Common interface and intentionally different responses

Both sensor types use the same sequence: construct local sampling points, read current body and geometry poses, transform each point into world coordinates, query nearby geometry, and produce a normalized response. The sensor adapter integrates this sequence with mjlab's initialization, update, cache, and reset lifecycle.

The common tensor has shape `[E, P, N, 4]`: environments, patches, points, and `(x_world, y_world, z_world, response)`. Distances are stored separately as `[E, P, N]`. Hands use N = 384 and feet use N = 1,024; both have four patches. The layout and sampling interface are unified while grid resolution and response parameters remain explicit.

For a local point p, `p_world = x_body + R_body * p_local`. Before a geometry query, `q = transpose(R_geom) * (p_world - x_geom)`.

The finger response follows the Week 2 thin-shell definition:

```text
r_hand = clip(5000 * max(0.0015 - d, 0) / 2.4, 0, 1)
```

Triangle-mesh distances are unsigned. Supported analytic primitives can return signed distances. The foot response uses no positive shell and a 2 mm normalization depth:

```text
r_foot = clip(max(-d, 0) / 0.002, 0, 1)
```

Directly applying the hand shell to a continuously grounded foot can saturate the readout. Foot depth normalization retains visible changes in the contact distribution. Hand colors use a fixed 0-1 scale; the four foot plots share a dynamic display scale. Raw statistics and saved arrays retain the original response values. Neither mapping is a hardware pressure calibration.

## FlexiTac methods and mjlab adaptations

The implementation reproduces the regular surface sampling, rigid-body coordinate transforms, Warp triangle-nearest-point queries, and distance-to-response structure used in the FlexiTac Isaac implementation. The relevant Warp operations are `wp.Mesh`, `wp.mesh_query_point`, and `wp.mesh_eval_position`.

The Isaac-specific `WarpSdfTactileSensor` class is not imported. USD mesh extraction, PhysX pose access, and Isaac sensor callbacks are replaced with MuJoCo model data, MuJoCo Warp poses, and mjlab Sensor adapters. The current implementation is therefore a method reproduction and port, with additional scene-query support, rather than direct use of the original Isaac sensor class.

The current query merges opposing finger meshes and eligible scene geometry. It supports planes, spheres, capsules, cylinders, boxes, ellipsoids, and triangle meshes. Ellipsoid distance is approximate. Triangle meshes use unsigned nearest-surface distances rather than the original signed winding-number workflow. Heightfields and custom SDF geometry are not implemented. Geometry candidates are discovered when the scene is built; adding new geometry requires rebuilding the scene.

The system does not simulate deformable skin, electrical outputs, or calibrated physical pressure. Sampling responses do not independently apply spring forces to the physical solver.

## Findings during migration

### Right-hand contact-plane alignment

The initial port used the wrong local inward signs for the two right fingers. Their tactile sampling planes were 1 mm behind the actual ball-facing collision-pad surfaces, while the left sampling planes already matched those surfaces. The corrected right-bottom and right-top signs are -1 and +1. This correction changes the sensing planes and grid storage order. The physical pad geometry, sphere mass/radius, contact parameters, and controller targets stay the same. The display still places the finger root at the top.

The dual-grasp regression now independently derives each ball-facing box surface from the model, verifies all sampling points lie on it, and checks display orientation. In the corrected default grasp, every finger has 42 active taxels out of 384 and mean response approximately 0.0853. The largest contact-plane placement error is below 2e-9 m. The corrected image and raw values are in `results/ball_grasp/dual_held_heatmaps.*`, and `dual_check.json` includes `tactile_contact_planes` and `initial_tactile`. Earlier screenshots and screen recordings document the implementation before this correction.

### Continuous sensing after release

Restricting a sensor to a named ball loses finger-to-finger proximity when the ball is released. Each update now includes the opposing finger and supported scene objects, so an empty closing grasp can still produce a response. Signals clear when surfaces move sufficiently far away.

### Visual collision geometry and tactile geometry

Convex collision hulls of concave finger meshes can fill the intended gripper cavity. The grasp scene therefore uses four 24 x 64 x 1 mm planar pads as the effective finger contact surfaces. Tactile mesh queries still have access to full triangle surfaces. Physical collision proxies and sampling geometry serve different purposes and should not be treated as identical models.

### Image orientation

Finger arrays are stored along local X and signed Z axes. The display transposes the storage grid and corrects the opposing pad orientation so all finger images appear upright, with root at the top and tip at the bottom. This is a display transformation; it does not rotate the physical sensor or change raw observations. Foot images place the toe at the top and local +X to the right.

### Unequal initial foot signals

The source URDF has approximately 0.820 mm of zero-pose sole height difference: a 0.766 mm difference in hip-roll vertical joint origins plus a 0.054 mm difference in ankle-pitch origins. The CAD design reason is not established. The complete demo lowers the robot's initial root height by 6 mm so the soles participate in ground contact; it does not independently correct the leg origins or edit the STL geometry.

Foot tilt controls set `l_ankle_roll` and `r_ankle_roll` position targets. They rotate downstream foot bodies and sampling points through the joint chain. They do not permanently modify model installation transforms. Actual joint angles may differ from their targets under contact and load. A fixed-base robot also has constraint forces, so the summed foot response cannot be interpreted as total body weight.

## Validation evidence

The saved foot check uses four environments and reports tensor shape `[4, 4, 1024, 4]`, maximum world-position error approximately 1.19e-7 m, and maximum sole-surface placement error approximately 0.055 mm. Tilt changes, reset behavior, and finite observations are checked.

The saved dual-ball check covers simultaneous holding, separate release, empty closing, changed ball offsets, and per-hand reset isolation. The recorded four-second grasp drift was approximately 0.477 mm per ball. Per-hand vertical contact forces sum to approximately 0.3924 N, consistent with the 40 g ball's weight. Object probes cover six geometry classes on all four finger arrays, including response at 1.2 mm surface distance and clearing after withdrawal.

The full observation contains 81 base values, 1,536 finger responses, 18 ball-state values, and 4,096 foot responses, totaling 5,731 values. There are 5,632 tactile sampling points and 27 actions.

Historical evidence is retained under `results`. Cleanup and launcher checks write their own reports under `results/validation`; these are separate from the earlier measured test records.

## Python file roles

The package contains eight core modules in `code/src/sunday_a1_mjlab`:

| File | Responsibility and action involvement |
| --- | --- |
| `__init__.py` | Package description; no motion or sampling. |
| `tasks.py` | Registers JointCheck, FlexiTac, and BallGrasp tasks; defines the model, position actuators, initial state, actions, observations, and rewards. |
| `ball_grasp.py` | Creates the two free balls and planar contact pads; adds contact and foot sensors; `reset_hand` writes only one ball and that hand's two finger states. |
| `tactile_backend.py` | Creates the mjlab environment; converts finger targets to actions, applies ankle targets, advances physics, and synchronizes the display copy. |
| `flexitac.py` | Common Warp grid engine, distance queries, scene-geometry merging, and response data. It reads poses and does not command joints. |
| `feet_tactile.py` | Defines the four sole patches and foot grid/response parameters using the common sampling engine. |
| `mjlab_flexitac.py` | Implements mjlab Sensor lifecycle adapters and hand/foot observation functions. |
| `tactile_display.py` | Viser controls, model display, correctly oriented heatmaps, and PNG/NPZ/JSON export. Control events are consumed by the main loop. |

The thirteen scripts are in `code/scripts`:

| File | Responsibility |
| --- | --- |
| `ball_grasp_demo.py` | Main two-ball/eight-array interface and control loop. |
| `convert_robot.py` | Converts the included URDF/STL model to runtime MJCF assets. |
| `preview_robot.py` | Basic native MuJoCo model preview with independent L/R gripper control; no complete ball scene. |
| `render_preview.py` | Produces a static MuJoCo model image without advancing physics. |
| `flexitac_demo.py` | Earlier four-finger open/close interface, with mjlab or native MuJoCo backend. |
| `record_flexitac_demo.py` | Records four finger open/close combinations and their heatmaps. |
| `record_ball_grasp.py` | Earlier left-hand grasp, release, and whole-environment-reset video; its layout does not include foot heatmaps. |
| `check_mjlab.py` | Checks environment registration, GPU execution, parallel-world separation, and finite values. |
| `check_flexitac_mjlab.py` | Checks independent finger states, wrist rotation, point transforms, cached reads, and reset. |
| `check_ball_grasp.py` | Earlier single-ball hold/release check. |
| `check_dual_ball_grasp.py` | Checks both grasps, separate release, empty close, offsets, real forces, and reset isolation. |
| `check_tactile_objects.py` | Tests six supported object classes moving near and away from the four finger arrays. |
| `check_feet_tactile.py` | Checks foot layouts, surface placement, transforms, tilt response, reset, and observations. |

`code/run_demo.py` selects the existing compatible Python dependencies, registers the local task package, and dispatches these scripts. `code/launch.ps1` locates the existing Python environment. The batch launchers provide convenient Windows entry points. External source links are listed only in the root README.
