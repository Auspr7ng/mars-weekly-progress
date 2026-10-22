"""Install four Sunday A1 finger FlexiTac sensors in the open Isaac Sim stage.

Run first from Window > Script Editor with the timeline stopped and the corrected
Sunday A1 URDF imported at /World/sunday_assembly. Rerunning on the same stage
refreshes the helper functions without duplicating sensors. The script adds four thin
colored placement guides to the moving finger links, then creates four separate
WarpSdfTactileSensor instances. Press Play to initialize and sample them.

This first contact test queries the *opposing finger mesh* for each sensor. The
output is an unsigned SDF proximity/contact proxy, not calibrated contact force.
The script opens a live 2x2 heatmap with per-finger numeric summaries.
Call builtins.sunday_finger_status() for four independent readings, and
builtins.sunday_finger_heatmaps() for a one-time 2x2 PNG. Call
builtins.set_sunday_grippers(0.03) / (0.0) to open / close both grippers, or
builtins.sunday_demo_cycle() to hold, open, hold, close, and diagnose.
"""

from __future__ import annotations

import builtins
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
import torch
from omni.kit.async_engine import run_coroutine
from pxr import Gf, UsdGeom, UsdPhysics

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.simulation_context import SimulationContext
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas import activate_contact_sensors
from isaaclab.sensors.warp_sdf_tactile import WarpSdfTactileSensor, WarpSdfTactileSensorCfg


ROBOT = "/World/sunday_assembly"
CALLBACK = "sunday_four_finger_flexitac"
TAXEL_VIEW = "/World/SundayFlexiTacTaxels"
OUTPUT = Path("D:/robotics/SundayA1_FlexiTac/output/four_finger_heatmaps.png")
ROWS, COLS, PITCH_M = 12, 32, 0.002
PATCH_X_M, PATCH_Z_M = -0.134, -0.035
SURFACE_CLEARANCE_M = 0.0005
SDF_SHELL_M = 0.0015
SDF_MAX_DISTANCE_M = 0.10
SDF_STIFFNESS_PER_M = 5000.0
GRIPPER_DRIVE_STIFFNESS = 1000.0
GRIPPER_DRIVE_DAMPING = 30.0
RED_THRESHOLD = 0.01
VISUAL_PERIOD_S = 0.15
HEATMAP_PERIOD_S = 0.1
HEATMAP_VERSION = 2


@dataclass(frozen=True)
class Finger:
    name: str
    link: str
    opposite: str
    surface_y_m: float
    inward_sign: int
    color: tuple[float, float, float]

    @property
    def path(self) -> str:
        return f"{ROBOT}/{self.link}"

    @property
    def patch_pos(self) -> tuple[float, float, float]:
        return (
            PATCH_X_M,
            self.surface_y_m + self.inward_sign * SURFACE_CLEARANCE_M,
            PATCH_Z_M,
        )

    @property
    def patch_quat(self) -> tuple[float, float, float, float]:
        # The patch-local +Y axis points into the gripper opening.
        return (1.0, 0.0, 0.0, 0.0) if self.inward_sign > 0 else (0.0, 1.0, 0.0, 0.0)


# Largest inward-facing planar STL surfaces, measured in each URDF link frame.
FINGERS = (
    Finger("left_bottom", "l_finger_bottom_1", "l_finger_top_1", -0.0494, -1, (0.1, 0.9, 0.2)),
    Finger("left_top", "l_finger_top_1", "l_finger_bottom_1", 0.0289, +1, (0.1, 0.8, 0.9)),
    Finger("right_bottom", "r_finger_bottom_1", "r_finger_top_1", 0.0488, +1, (0.9, 0.8, 0.1)),
    Finger("right_top", "r_finger_top_1", "r_finger_bottom_1", -0.0309, -1, (0.9, 0.2, 0.8)),
)
FINGER_BY_LINK = {finger.link: finger for finger in FINGERS}
JOINT_NAMES = (
    "l_finger_slide_bottom",
    "l_finger_slide_top",
    "r_finger_slide_bottom",
    "r_finger_slide_top",
)


def find_finger_mesh(link_path: str) -> str:
    """Resolve a real USD Mesh prim, preferring the imported collision mesh."""
    matches = sim_utils.get_all_matching_child_prims(
        link_path,
        predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        traverse_instance_prims=True,
    )
    paths = [prim.GetPath().pathString for prim in matches]
    for part in ("/collisions/", "/visuals/"):
        selected = [path for path in paths if part in path.lower()]
        if selected:
            return selected[0]
    raise RuntimeError(f"No USD Mesh prim found under {link_path}. Found: {paths}")


def target_mesh_scale(stage, path: str) -> float:
    """Get the uniform world scale used by the Warp target mesh."""
    scale_xyz = tuple(float(value) for value in sim_utils.resolve_prim_scale(stage.GetPrimAtPath(path)))
    if min(scale_xyz) <= 0 or max(scale_xyz) / min(scale_xyz) > 1.01:
        raise RuntimeError(f"Target mesh needs positive uniform scale: {path} has {scale_xyz}")
    return sum(scale_xyz) / 3.0


def configure_sensor_mesh_units(sensor: WarpSdfTactileSensor, scale: float) -> None:
    """Warp reports SDF distances in mesh-local units, not world meters."""
    sensor.cfg.mesh_max_dist = SDF_MAX_DISTANCE_M / scale
    sensor.cfg.mesh_shell_thickness = SDF_SHELL_M / scale
    sensor.cfg.stiffness = SDF_STIFFNESS_PER_M * scale


def target_mesh_local_poses(stage, target_paths: dict[str, str]) -> dict[str, tuple[tuple, tuple]]:
    """Cache each target mesh pose relative to its moving finger rigid body."""
    poses = {}
    for finger in FINGERS:
        mesh_prim = stage.GetPrimAtPath(target_paths[finger.name])
        body_prim = stage.GetPrimAtPath(f"{ROBOT}/{finger.opposite}")
        pos_b, quat_b = sim_utils.resolve_prim_pose(mesh_prim, ref_prim=body_prim)
        poses[finger.name] = (tuple(float(value) for value in pos_b), tuple(float(value) for value in quat_b))
    return poses


def add_placement_guide(stage, finger: Finger) -> None:
    """A visible, non-colliding patch that inherits the finger transform."""
    path = f"{finger.path}/FlexiTacPlacementGuide"
    if stage.GetPrimAtPath(path).IsValid():
        return
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*finger.patch_pos))
    cube.AddScaleOp().Set(Gf.Vec3f(0.024, 0.0003, 0.064))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*finger.color)])
    cube.CreateDisplayOpacityAttr([0.55])


def get_gripper_articulation(stage) -> tuple[SingleArticulation, tuple[int, ...]]:
    """Get a PhysX tensor-backed articulation handle for the active scene."""
    roots = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if (prim.GetPath().pathString == ROBOT or prim.GetPath().pathString.startswith(ROBOT + "/"))
        and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation root under {ROBOT}, found {roots}.")
    articulation = getattr(builtins, "sunday_gripper_articulation", None)
    if (
        articulation is None
        or getattr(builtins, "sunday_gripper_articulation_stage", None) != stage
        or articulation.prim_path != roots[0]
    ):
        articulation = SingleArticulation(
            prim_path=roots[0], name="sunday_finger_control", reset_xform_properties=False
        )
        builtins.sunday_gripper_articulation = articulation
        builtins.sunday_gripper_articulation_stage = stage
    if not articulation.handles_initialized:
        articulation.initialize()
    indices = tuple(articulation.get_dof_index(name) for name in JOINT_NAMES)
    if any(index is None for index in indices) or len(set(indices)) != 4:
        raise RuntimeError(f"Could not identify four distinct finger DOFs: {dict(zip(JOINT_NAMES, indices))}")
    return articulation, indices


def configure_sunday_gripper_drives(stage=None) -> None:
    """Increase only the four finger PD gains, without changing other joints."""
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    if stage is None or not stage.GetPrimAtPath(ROBOT).IsValid():
        raise RuntimeError(f"Current stage has no robot at {ROBOT}.")
    if omni.timeline.get_timeline_interface().is_stopped():
        for name in JOINT_NAMES:
            prim = stage.GetPrimAtPath(f"{ROBOT}/joints/{name}")
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
            if not drive:
                raise RuntimeError(f"Linear drive missing: {name}")
            drive.GetStiffnessAttr().Set(GRIPPER_DRIVE_STIFFNESS)
            drive.GetDampingAttr().Set(GRIPPER_DRIVE_DAMPING)
        print(f"Finger USD drive gains: Kp={GRIPPER_DRIVE_STIFFNESS:g}, Kd={GRIPPER_DRIVE_DAMPING:g}")
        return

    # USD drive writes are rejected by GPU PhysX while Play/Pause is active.
    articulation, indices = get_gripper_articulation(stage)
    controller = articulation.get_articulation_controller()
    kps, kds = controller.get_gains()
    if kps is None or kds is None:
        raise RuntimeError("Could not read live articulation drive gains.")
    current = [(float(kps[index]), float(kds[index])) for index in indices]
    if all(abs(kp - GRIPPER_DRIVE_STIFFNESS) < 1e-3 and abs(kd - GRIPPER_DRIVE_DAMPING) < 1e-3
           for kp, kd in current):
        return
    kps = kps.clone() if isinstance(kps, torch.Tensor) else kps.copy()
    kds = kds.clone() if isinstance(kds, torch.Tensor) else kds.copy()
    for index in indices:
        kps[index] = GRIPPER_DRIVE_STIFFNESS
        kds[index] = GRIPPER_DRIVE_DAMPING
    controller.set_gains(kps=kps, kds=kds)
    actual_kps, actual_kds = controller.get_gains()
    print("Finger live drive gains:", {
        name: (float(actual_kps[index]), float(actual_kds[index]))
        for name, index in zip(JOINT_NAMES, indices, strict=True)
    })


def set_sunday_grippers(position_m: float) -> None:
    """Command all finger slides; use PhysX tensors while Play/Pause is active."""
    position_m = float(position_m)
    if not 0.0 <= position_m <= 0.03:
        raise ValueError("Finger joint position must be between 0 and 0.03 m.")
    stage = omni.usd.get_context().get_stage()
    if stage is None or not stage.GetPrimAtPath(ROBOT).IsValid():
        raise RuntimeError(f"Current stage has no robot at {ROBOT}.")
    if not omni.timeline.get_timeline_interface().is_stopped():
        articulation, indices = get_gripper_articulation(stage)
        configure_sunday_gripper_drives(stage)
        joint_indices = torch.tensor(indices, dtype=torch.long)
        articulation.apply_action(
            ArticulationAction(
                joint_positions=torch.full((4,), position_m, dtype=torch.float32),
                joint_indices=joint_indices,
            )
        )
        print(f"PhysX finger target: {position_m:.4f} m for {dict(zip(JOINT_NAMES, indices))}")
        return

    # The USD drive attributes are only safe to write with the timeline stopped.
    configure_sunday_gripper_drives(stage)
    drives = []
    for joint_name in JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"{ROBOT}/joints/{joint_name}")
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            raise RuntimeError(f"Finger joint missing: {joint_name}")
        drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        if not drive:
            raise RuntimeError(f"Linear drive missing: {joint_name}")
        drives.append((joint_name, drive))
    for name, drive in drives:
        drive.GetTargetPositionAttr().Set(position_m)
        print(f"{name}: target={drive.GetTargetPositionAttr().Get():.4f} m")


def sunday_gripper_joint_positions() -> dict[str, float]:
    """Read actual slide positions, not just the requested drive targets."""
    if omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError("Press Play before reading live finger joint positions.")
    stage = omni.usd.get_context().get_stage()
    articulation, indices = get_gripper_articulation(stage)
    values = articulation.get_joint_positions(joint_indices=torch.tensor(indices, dtype=torch.long))
    result = {name: float(value) for name, value in zip(JOINT_NAMES, values, strict=True)}
    for name, value in result.items():
        print(f"{name}: actual={value:.4f} m")
    return result


builtins.set_sunday_grippers = set_sunday_grippers
builtins.configure_sunday_gripper_drives = configure_sunday_gripper_drives
builtins.sunday_gripper_joint_positions = sunday_gripper_joint_positions


def update_targets_and_sensors(dt: float) -> int:
    stage = omni.usd.get_context().get_stage()
    if stage != getattr(builtins, "sunday_flexitac_stage", None):
        builtins.sunday_last_sensor_updates = 0
        builtins.sunday_last_sensor_skips = ["stage changed"]
        return 0
    sensors = getattr(builtins, "sunday_finger_sensors", {})
    local_poses = getattr(builtins, "sunday_target_mesh_local_poses", {})
    tensor_cache = getattr(builtins, "sunday_target_mesh_local_tensors", {})
    updated = 0
    skipped = []
    for finger in FINGERS:
        sensor = sensors.get(finger.name)
        if sensor is None or not sensor.is_initialized:
            skipped.append(f"{finger.name}: tactile sensor not initialized")
            continue
        opposite = FINGER_BY_LINK[finger.opposite]
        opposite_sensor = sensors.get(opposite.name)
        if opposite_sensor is None or not opposite_sensor.is_initialized:
            skipped.append(f"{finger.name}: opposite tactile sensor not initialized")
            continue
        pose_sensor = opposite_sensor._pose_sensors[0]
        if not pose_sensor.is_initialized:
            skipped.append(f"{finger.name}: opposite pose sensor not initialized")
            continue
        pose_sensor.update(0.0, force_recompute=True)
        if pose_sensor.data.pos_w is None or pose_sensor.data.quat_w is None:
            skipped.append(f"{finger.name}: opposite body pose unavailable")
            continue
        body_pos = pose_sensor.data.pos_w[0, 0]
        body_quat = pose_sensor.data.quat_w[0, 0]
        cached = tensor_cache.get(finger.name)
        if cached is None or cached[0].device != body_pos.device:
            relative_pos, relative_quat = local_poses[finger.name]
            cached = (
                torch.tensor(relative_pos, device=body_pos.device, dtype=body_pos.dtype),
                torch.tensor(relative_quat, device=body_pos.device, dtype=body_quat.dtype),
            )
            tensor_cache[finger.name] = cached
        offset_pos, offset_quat = cached
        target_pos = body_pos + math_utils.quat_apply(body_quat.unsqueeze(0), offset_pos.unsqueeze(0))[0]
        target_quat = math_utils.quat_mul(body_quat.unsqueeze(0), offset_quat.unsqueeze(0))[0]
        sensor.set_target_pose(target_pos, target_quat)
        sensor.update(dt, force_recompute=True)
        updated += 1
    builtins.sunday_last_sensor_updates = updated
    builtins.sunday_last_sensor_skips = skipped
    return updated


def sunday_finger_readings() -> dict[str, torch.Tensor]:
    """Return independent (384, 4) tensors: world XYZ and normalized response."""
    sensors = getattr(builtins, "sunday_finger_sensors", {})
    if len(sensors) != 4 or not all(sensor.is_initialized for sensor in sensors.values()):
        raise RuntimeError("Four sensors are not initialized yet. Press Play once.")
    if update_targets_and_sensors(0.0) != 4:
        raise RuntimeError(f"Some sensors did not update: {getattr(builtins, 'sunday_last_sensor_skips', [])}")
    return {
        finger.name: sensors[finger.name].data.tactile_points_w_per_sensor[0, 0].clone()
        for finger in FINGERS
    }


def sunday_finger_status() -> dict[str, dict[str, float]]:
    """Print per-finger max response and number of active taxels."""
    result = {}
    for name, points in sunday_finger_readings().items():
        values = points[:, 3]
        result[name] = {
            "max_response": float(values.max().item()),
            "active_taxels": int((values > RED_THRESHOLD).sum().item()),
        }
        print(f"{name:>13}: max={result[name]['max_response']:.3f}, active={result[name]['active_taxels']}/{ROWS * COLS}")
    return result


def sunday_finger_heatmaps() -> str:
    """Save four independently labelled, normalized SDF-response heatmaps."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    readings = sunday_finger_readings()
    fig = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for axis, finger in zip(axes.flat, FINGERS, strict=True):
        values = readings[finger.name][:, 3].detach().cpu().numpy().reshape(ROWS, COLS)
        image = axis.imshow(values, origin="lower", cmap="jet", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_title(f"{finger.name}: max={values.max():.3f}")
        axis.set_xlabel("z taxel")
        axis.set_ylabel("x taxel")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Sunday A1 finger SDF response (normalized; not calibrated pressure)")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    print("Four heatmaps saved:", OUTPUT)
    return str(OUTPUT)


builtins.sunday_finger_readings = sunday_finger_readings
builtins.sunday_finger_status = sunday_finger_status
builtins.sunday_finger_heatmaps = sunday_finger_heatmaps


def sunday_debug_snapshot(label: str = "snapshot") -> dict:
    """Compare commanded/actual joints and closest taxel-to-mesh distances."""
    stage = omni.usd.get_context().get_stage()
    if stage is None or omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError("Press Play before collecting a gripper/sensor snapshot.")
    articulation, indices = get_gripper_articulation(stage)
    actual = articulation.get_joint_positions(joint_indices=torch.tensor(indices, dtype=torch.long))
    applied = articulation.get_applied_action()
    targets = applied.joint_positions if applied is not None else None
    kps, kds = articulation.get_articulation_controller().get_gains()
    result = {"joints": {}, "sensors": {}}
    print(f"--- {label} | physics steps={getattr(builtins, 'sunday_flexitac_steps', 0)} ---")
    for name, index, position in zip(JOINT_NAMES, indices, actual, strict=True):
        target = float(targets[index]) if targets is not None else float("nan")
        result["joints"][name] = {
            "target_m": target, "actual_m": float(position),
            "stiffness": float(kps[index]), "damping": float(kds[index]),
        }
        print(
            f"{name}: target={target:.4f} m, actual={float(position):.4f} m, "
            f"Kp={float(kps[index]):.1f}, Kd={float(kds[index]):.1f}"
        )

    sensors = getattr(builtins, "sunday_finger_sensors", {})
    scales = getattr(builtins, "sunday_finger_target_scales", {})
    for finger in FINGERS:
        sensor = sensors.get(finger.name)
        if sensor is None or not sensor.is_initialized or sensor._sdf_out is None:
            print(f"{finger.name}: sensor not initialized")
            continue
        values = sensor.data.tactile_points_w_per_sensor[0, 0, :, 3]
        taxel_xyz = sensor.data.tactile_points_w_per_sensor[0, 0, :, :3].mean(dim=0)
        mesh_xyz = sensor._mesh_pos_w
        scale = scales.get(finger.name, 1.0)
        closest_mm = float(sensor._sdf_out[0, 0].min().item()) * scale * 1000.0
        maximum = float(values.max().item())
        active = int((values > RED_THRESHOLD).sum().item())
        result["sensors"][finger.name] = {
            "closest_target_mm": closest_mm,
            "max_response": maximum,
            "active_taxels": active,
            "taxel_center_w": tuple(float(value) for value in taxel_xyz),
            "target_mesh_origin_w": tuple(float(value) for value in mesh_xyz) if mesh_xyz is not None else None,
        }
        print(f"{finger.name}: closest={closest_mm:.2f} mm, max={maximum:.3f}, active={active}/{ROWS * COLS}")
        print(
            f"  taxel center XYZ={result['sensors'][finger.name]['taxel_center_w']}, "
            f"target mesh origin XYZ={result['sensors'][finger.name]['target_mesh_origin_w']}"
        )
    print("sensor update error:", getattr(builtins, "sunday_flexitac_error", None))
    print(
        "last sensor updates:", getattr(builtins, "sunday_last_sensor_updates", None),
        "skipped:", getattr(builtins, "sunday_last_sensor_skips", None),
    )
    return result


def sunday_demo_cycle(initial_hold_s: float = 3.0, open_hold_s: float = 3.0, closed_hold_s: float = 3.0) -> None:
    """Non-blocking: wait, open and hold, then close and hold for live inspection."""
    durations = tuple(float(value) for value in (initial_hold_s, open_hold_s, closed_hold_s))
    if any(value < 0.0 or value > 60.0 for value in durations):
        raise ValueError("Each hold duration must be between 0 and 60 seconds.")
    sim = SimulationContext.instance()
    if sim is None or not sim.is_playing():
        raise RuntimeError("Press Play before starting the timed demo cycle.")
    if getattr(builtins, "sunday_demo_running", False):
        raise RuntimeError("A Sunday gripper demo cycle is already running.")
    builtins.sunday_demo_running = True
    builtins.sunday_demo_error = None

    async def run_cycle() -> None:
        try:
            print(f"Holding the current pose for {durations[0]:.1f} s before motion...")
            await asyncio.sleep(durations[0])
            if not sim.is_playing():
                raise RuntimeError("The timeline stopped or paused during the initial hold.")
            sunday_debug_snapshot("after initial hold, before motion")
            set_sunday_grippers(0.03)
            print(f"Holding the open command for {durations[1]:.1f} s...")
            await asyncio.sleep(durations[1])
            if not sim.is_playing():
                raise RuntimeError("The timeline stopped or paused during the open hold.")
            sunday_debug_snapshot("after open hold")
            set_sunday_grippers(0.0)
            print(f"Holding the close command for {durations[2]:.1f} s...")
            await asyncio.sleep(durations[2])
            if not sim.is_playing():
                raise RuntimeError("The timeline stopped or paused during the close hold.")
            sunday_debug_snapshot("after close hold")
            print("Sunday gripper demo cycle finished.")
        except Exception as exc:
            builtins.sunday_demo_error = repr(exc)
            print("Sunday gripper demo error:", repr(exc))
        finally:
            builtins.sunday_demo_running = False

    builtins.sunday_demo_task = run_coroutine(run_cycle())


builtins.sunday_debug_snapshot = sunday_debug_snapshot
builtins.sunday_demo_cycle = sunday_demo_cycle


class SundayFourFingerLiveHeatmap:
    """Show four latest 12x32 SDF grids and numeric summaries inside Isaac Sim."""

    def __init__(self, sim: SimulationContext, stage) -> None:
        self.sim = sim
        self.stage = stage
        self.version = HEATMAP_VERSION
        self.last_update = 0.0
        self.update_count = 0
        self.error = None
        self.providers = {finger.name: ui.ByteImageProvider() for finger in FINGERS}
        self.image_frames = {}
        self.value_labels = {}

        blank = self._rgba_image(np.zeros((ROWS, COLS), dtype=np.float32))
        for provider in self.providers.values():
            provider.set_bytes_data(bytearray(blank.tobytes()), [blank.shape[1], blank.shape[0]])

        self.window = ui.Window("Sunday A1 | Four FlexiTac Heatmaps", width=760, height=440)
        with self.window.frame:
            with ui.VStack(spacing=7):
                self.status_label = ui.Label("Press Play | normalized SDF response, not pressure")
                for row in (FINGERS[:2], FINGERS[2:]):
                    with ui.HStack(spacing=16):
                        for finger in row:
                            with ui.VStack(spacing=4):
                                ui.Label(finger.name)
                                self.value_labels[finger.name] = ui.Label(
                                    f"max 0.000 | mean 0.000 | active 0/{ROWS * COLS}"
                                )
                                frame = ui.Frame(width=320, height=120)
                                provider = self.providers[finger.name]
                                frame.set_build_fn(
                                    lambda provider=provider: ui.ImageWithProvider(
                                        provider,
                                        width=320,
                                        height=120,
                                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                                    )
                                )
                                frame.rebuild()
                                self.image_frames[finger.name] = frame
                ui.Label("Blue = 0, red = 1; fixed scale shared by all four fingers")

        stream = omni.kit.app.get_app_interface().get_post_update_event_stream()
        self.subscription = stream.create_subscription_to_pop(self._on_update)

    @staticmethod
    def _rgba_image(grid: np.ndarray) -> np.ndarray:
        """Convert a normalized grid into a blue-to-red RGBA image."""
        intensity = np.clip(np.flipud(grid), 0.0, 1.0)
        rgba = np.empty((*intensity.shape, 4), dtype=np.uint8)
        rgba[..., 0] = (255.0 * np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0.0, 1.0)).astype(np.uint8)
        rgba[..., 1] = (255.0 * np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0.0, 1.0)).astype(np.uint8)
        rgba[..., 2] = (255.0 * np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0.0, 1.0)).astype(np.uint8)
        rgba[..., 3] = 255
        return np.repeat(np.repeat(rgba, 10, axis=0), 10, axis=1)

    def _on_update(self, event) -> None:
        if self.error is not None or not self.window.visible:
            return
        if omni.usd.get_context().get_stage() != self.stage:
            return
        if not self.sim.is_playing():
            if self.update_count:
                self.status_label.text = f"Paused | last live frame {self.update_count}"
            return
        now = time.monotonic()
        if now - self.last_update < HEATMAP_PERIOD_S:
            return
        sensors = getattr(builtins, "sunday_finger_sensors", {})
        if len(sensors) != 4 or not all(sensor.is_initialized for sensor in sensors.values()):
            return

        try:
            summaries = {}
            for finger in FINGERS:
                tensor = sensors[finger.name].data.tactile_points_w_per_sensor
                grid = tensor[0, 0, :, 3].detach().cpu().numpy().reshape(ROWS, COLS)
                grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
                image = self._rgba_image(grid)
                self.providers[finger.name].set_bytes_data(
                    bytearray(image.tobytes()), [image.shape[1], image.shape[0]]
                )
                self.image_frames[finger.name].rebuild()
                maximum = float(grid.max())
                mean = float(grid.mean())
                active = int(np.count_nonzero(grid > RED_THRESHOLD))
                summaries[finger.name] = {"max_response": maximum, "mean_response": mean, "active_taxels": active}
                self.value_labels[finger.name].text = (
                    f"max {maximum:.3f} | mean {mean:.3f} | active {active}/{ROWS * COLS}"
                )

            self.update_count += 1
            self.last_update = now
            builtins.sunday_finger_live_data = summaries
            builtins.sunday_heatmap_update_count = self.update_count
            self.status_label.text = (
                f"Playing | physics steps {getattr(builtins, 'sunday_flexitac_steps', 0)}"
                f" | heatmap frames {self.update_count} | normalized SDF, not pressure"
            )
        except Exception as exc:
            self.error = repr(exc)
            builtins.sunday_heatmap_error = self.error
            print("Sunday four-finger live heatmap error:", self.error)

    def close(self) -> None:
        self.subscription = None
        self.window.visible = False


def ensure_live_heatmap(sim: SimulationContext, stage) -> None:
    existing = getattr(builtins, "sunday_live_heatmap", None)
    if (
        existing is not None
        and getattr(existing, "stage", None) == stage
        and getattr(existing, "version", None) == HEATMAP_VERSION
    ):
        existing.window.visible = True
        return
    if existing is not None:
        existing.close()
    builtins.sunday_heatmap_error = None
    builtins.sunday_heatmap_update_count = 0
    builtins.sunday_finger_live_data = {}
    builtins.sunday_live_heatmap = SundayFourFingerLiveHeatmap(sim, stage)


def show_sunday_live_heatmap() -> None:
    """Reopen the live window if it was hidden."""
    sim = SimulationContext.instance()
    stage = omni.usd.get_context().get_stage()
    if sim is None or stage is None:
        raise RuntimeError("The four-finger simulation is not set up yet.")
    ensure_live_heatmap(sim, stage)


builtins.show_sunday_live_heatmap = show_sunday_live_heatmap


def ensure_live_callbacks(sim: SimulationContext, stage) -> None:
    ensure_live_heatmap(sim, stage)
    # Replace a callback from an older Script Editor run so the latest pose logic is used.
    if sim.physics_callback_exists(CALLBACK):
        sim.remove_physics_callback(CALLBACK)
    builtins.sunday_flexitac_steps = 0
    builtins.sunday_flexitac_error = None

    def physics_step(dt):
        if builtins.sunday_flexitac_error is not None:
            return
        try:
            update_targets_and_sensors(dt)
            builtins.sunday_flexitac_steps += 1
        except Exception as exc:
            builtins.sunday_flexitac_error = repr(exc)
            print("Sunday FlexiTac physics update error:", repr(exc))

    sim.add_physics_callback(CALLBACK, physics_step)

    view = getattr(builtins, "sunday_taxel_view", None)
    if view is None or view.stage != stage or not stage.GetPrimAtPath(view.prim_path).IsValid():
        view = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=TAXEL_VIEW,
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=0.0011,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "free": sim_utils.SphereCfg(
                        radius=0.0011,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
                    ),
                },
            )
        )
        view.set_visibility(False)
        builtins.sunday_taxel_view = view

    # Drop a stale post-update closure when the source file is re-executed.
    builtins.sunday_taxel_subscription = None
    builtins.sunday_taxel_last = 0.0
    builtins.sunday_taxel_error = None

    def refresh_taxels(event):
        if builtins.sunday_taxel_error is not None or not sim.is_playing():
            return
        if omni.usd.get_context().get_stage() != stage:
            return
        now = time.monotonic()
        if now - builtins.sunday_taxel_last < VISUAL_PERIOD_S:
            return
        try:
            sensors = getattr(builtins, "sunday_finger_sensors", {})
            if len(sensors) != 4 or not all(sensor.is_initialized for sensor in sensors.values()):
                return
            points = torch.cat(
                [sensors[finger.name].data.tactile_points_w_per_sensor[0, 0] for finger in FINGERS], dim=0
            )
            colors = (points[:, 3] <= RED_THRESHOLD).to(torch.int64)
            view.set_visibility(True)
            view.visualize(translations=points[:, :3], marker_indices=colors)
            builtins.sunday_taxel_last = now
        except RuntimeError:
            # Play callback may not have initialized all four sensors yet.
            return
        except Exception as exc:
            builtins.sunday_taxel_error = repr(exc)
            print("Sunday FlexiTac taxel visualization error:", repr(exc))

    stream = omni.kit.app.get_app_interface().get_post_update_event_stream()
    builtins.sunday_taxel_subscription = stream.create_subscription_to_pop(refresh_taxels)
    builtins.sunday_taxel_subscription_stage = stage


async def setup() -> None:
    if getattr(builtins, "sunday_flexitac_setup_running", False):
        print("Sunday FlexiTac setup is already running.")
        return
    builtins.sunday_flexitac_setup_running = True
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(ROBOT).IsValid():
            raise RuntimeError(f"Open a stage containing {ROBOT} first.")
        if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-6:
            raise RuntimeError("This configuration assumes stage units of 1 meter.")
        for finger in FINGERS:
            prim = stage.GetPrimAtPath(finger.path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Finger rigid body not found: {finger.path}")

        existing = getattr(builtins, "sunday_finger_sensors", None)
        if getattr(builtins, "sunday_flexitac_stage", None) == stage and existing and len(existing) == 4:
            configure_sunday_gripper_drives(stage)
            target_paths = getattr(builtins, "sunday_finger_targets", {})
            scales = {finger.name: target_mesh_scale(stage, target_paths[finger.name]) for finger in FINGERS}
            for finger in FINGERS:
                configure_sensor_mesh_units(existing[finger.name], scales[finger.name])
                print(f"{finger.name}: target mesh scale={scales[finger.name]:.6g}, shell={SDF_SHELL_M * 1000:.1f} mm")
            builtins.sunday_finger_target_scales = scales
            builtins.sunday_target_mesh_local_poses = target_mesh_local_poses(stage, target_paths)
            builtins.sunday_target_mesh_local_tensors = {}
            builtins.sunday_flexitac_error = None
            sim = SimulationContext.instance()
            if sim is not None and sim.get_physics_context() is not None:
                ensure_live_callbacks(sim, stage)
            print("Four existing sensors updated for target-mesh units; no duplicates created.")
            return

        physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        physics_scene = physics_scenes[0].GetPath().pathString if physics_scenes else "/World/PhysicsScene"
        if not physics_scenes:
            UsdPhysics.Scene.Define(stage, physics_scene)

        sim = SimulationContext.instance()
        if sim is None:
            sim = SimulationContext(
                physics_dt=1.0 / 120.0,
                rendering_dt=1.0 / 60.0,
                stage_units_in_meters=1.0,
                physics_prim_path=physics_scene,
                set_defaults=False,
                backend="torch",
                device="cuda:0",
                stage=stage,
            )
        if sim.is_playing():
            raise RuntimeError("Stop the timeline, then run this setup script once.")
        configure_sunday_gripper_drives(stage)
        if sim.get_physics_context() is None:
            await sim.initialize_simulation_context_async()
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError("Stage changed during setup. Reopen the robot stage and retry.")

        target_paths = {finger.name: find_finger_mesh(f"{ROBOT}/{finger.opposite}") for finger in FINGERS}
        scales = {finger.name: target_mesh_scale(stage, target_paths[finger.name]) for finger in FINGERS}
        for finger in FINGERS:
            add_placement_guide(stage, finger)
            activate_contact_sensors(finger.path, threshold=0.0)

        sensors = {}
        for finger in FINGERS:
            cfg = WarpSdfTactileSensorCfg(
                prim_path=ROBOT,
                elastomer_prim_paths=[finger.path],
                num_rows=ROWS,
                num_cols=COLS,
                point_distance=PITCH_M,
                normal_axis=1,
                normal_offset=0.0,
                patch_offset_pos_b=finger.patch_pos,
                patch_offset_quat_b=finger.patch_quat,
                target_mesh_prim_path=target_paths[finger.name],
                mesh_max_dist=SDF_MAX_DISTANCE_M / scales[finger.name],
                mesh_use_signed_distance=False,
                mesh_shell_thickness=SDF_SHELL_M / scales[finger.name],
                stiffness=SDF_STIFFNESS_PER_M * scales[finger.name],
                max_force=2.4,
                normalize_forces=True,
                debug_vis=False,
            )
            sensors[finger.name] = WarpSdfTactileSensor(cfg)
            print(
                f"{finger.name}: {finger.path} -> {target_paths[finger.name]} "
                f"(mesh scale {scales[finger.name]:.6g})"
            )

        builtins.sunday_flexitac_stage = stage
        builtins.sunday_finger_targets = target_paths
        builtins.sunday_finger_target_scales = scales
        builtins.sunday_target_mesh_local_poses = target_mesh_local_poses(stage, target_paths)
        builtins.sunday_target_mesh_local_tensors = {}
        builtins.sunday_finger_sensors = sensors
        ensure_live_callbacks(sim, stage)
        print("Four Sunday A1 finger sensors installed. Press Play, then call sunday_finger_status().")
    finally:
        builtins.sunday_flexitac_setup_running = False


run_coroutine(setup())
