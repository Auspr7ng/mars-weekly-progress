"""Attach the verified two-foot FlexiTac sensor in Isaac Sim 5.1.

Usage in Isaac Sim's Script Editor:
1. Open D:/robotics/flexitac_test.usd and leave the timeline stopped.
2. Open this file in Script Editor and run it once.
3. Wait for "FlexiTac two-foot setup ready" in the output.
4. Press Play, then Pause. The sensor initializes on Play and updates on each
   physics step. Red/blue taxel markers refresh up to 10 times per second while
   playing. A live two-foot pressure heatmap also opens in Isaac Sim and
   refreshes while Play is active. Check ``builtins.flexitac_step_count`` for
   completed updates.
5. Run ``builtins.read_flexitac_feet()`` to get the tactile tensor.
6. Run ``builtins.show_flexitac_pressure_heatmap()`` to save and open a
   one-time left/right heatmap of the latest physics step.

The USD stores the robot and ground, but the Python sensor must be recreated
after restarting Isaac Sim. Re-running this file on the same stage is safe.

The two taxel-plane offsets below were fitted to one static standing pose
(PhysX vertical forces about 14.4 N left and 17.9 N right). They are a
provisional aggregate-force calibration, not a validated pressure map.
"""

import builtins
import time

import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
import torch
from omni.kit.async_engine import run_coroutine
from pxr import UsdGeom, UsdPhysics

from isaacsim.core.api.simulation_context import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas import activate_contact_sensors
from isaaclab.sensors.warp_sdf_tactile import (
    WarpSdfTactileSensor,
    WarpSdfTactileSensorCfg,
)


ROBOT = "/World/sunday_assembly_step"
LEFT_FOOT = ROBOT + "/l_foot_1_1"
RIGHT_FOOT = ROBOT + "/r_foot_1_1"
GROUND = "/World/FlexiTacGround"
PHYSICS_SCENE = "/World/PhysicsScene"
LEFT_CALIBRATION_DOWN_M = 9.500755e-6
RIGHT_CALIBRATION_DOWN_M = 9.240334e-6
CALLBACK_NAME = "flexitac_feet_live"
RED_THRESHOLD = 0.01
VISUAL_PERIOD_S = 0.1
HEATMAP_PERIOD_S = 0.1
HEATMAP_VERSION = 3


def read_flexitac_feet():
    """Return (1, 2, 384, 4): environment, foot, taxel, [x, y, z, force]."""
    sensor = getattr(builtins, "flexitac_feet", None)
    if sensor is None or not sensor.is_initialized:
        raise RuntimeError("Press Play once to initialize the FlexiTac sensor.")
    sensor.update(0.0, force_recompute=True)
    return sensor.data.tactile_points_w_per_sensor


builtins.read_flexitac_feet = read_flexitac_feet


def show_flexitac_pressure_heatmap():
    """Save and open a one-time estimated-pressure heatmap for both feet."""
    import os

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sensor = getattr(builtins, "flexitac_feet", None)
    if sensor is None or not sensor.is_initialized:
        raise RuntimeError("Press Play once to initialize the FlexiTac sensor.")

    tactile = sensor.data.tactile_points_w_per_sensor
    if tactile is None or tactile.ndim != 4 or tactile.shape[1] != 2:
        raise RuntimeError("Expected tactile data with shape (environments, 2, points, 4).")

    values = tactile[0, :, :, 3].detach().cpu().numpy()
    forces = values * sensor.cfg.max_force if sensor.cfg.normalize_forces else values
    pressures = forces / (sensor.cfg.point_distance ** 2) / 1000.0

    fig = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    color_max = max(float(pressures.max()), 0.001)

    for index, name in enumerate(("Left foot", "Right foot")):
        image = axes[index].imshow(
            pressures[index].reshape(sensor.cfg.num_rows, sensor.cfg.num_cols),
            cmap="jet",
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=color_max,
        )
        axes[index].set_title(f"{name} | max {pressures[index].max():.3f} kPa")
        axes[index].set_xlabel("Column")
        axes[index].set_ylabel("Row")

    fig.colorbar(image, ax=axes, label="Estimated pressure (kPa)")
    output_path = "D:/robotics/flexitac_pressure_heatmap.png"
    fig.savefig(output_path, dpi=180)
    print("Heatmap saved to:", output_path)
    os.startfile(output_path)
    return output_path


builtins.show_flexitac_pressure_heatmap = show_flexitac_pressure_heatmap


def ensure_live_update(sim):
    """Compute fresh tactile values at every physics step, without USD writes."""
    if sim.physics_callback_exists(CALLBACK_NAME):
        return

    builtins.flexitac_step_count = 0
    builtins.flexitac_live_error = None

    def update_each_step(dt):
        if builtins.flexitac_live_error is not None:
            return
        sensor = getattr(builtins, "flexitac_feet", None)
        if sensor is None or not sensor.is_initialized:
            return
        try:
            sensor.update(dt, force_recompute=True)
            builtins.flexitac_step_count += 1
        except Exception as exc:
            builtins.flexitac_live_error = repr(exc)

    sim.add_physics_callback(CALLBACK_NAME, update_each_step)


def make_taxel_view(path):
    view = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=path,
            markers={
                "pressed": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "free": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
            },
        )
    )
    # PointInstancer prototypes start at the world origin. Hide them until
    # the first valid tactile data arrives so they do not appear near the hip.
    view.set_visibility(False)
    return view


def ensure_live_visualization(sim, stage):
    views = getattr(builtins, "flexitac_views", None)
    if views is None or len(views) != 2 or any(
        view.stage != stage or not stage.GetPrimAtPath(view.prim_path).IsValid() for view in views
    ):
        views = [
            make_taxel_view("/World/FlexiTac_View_Left"),
            make_taxel_view("/World/FlexiTac_View_Right"),
        ]
        builtins.flexitac_views = views

    if getattr(builtins, "flexitac_vis_subscription", None) is not None:
        return

    builtins.flexitac_vis_error = None
    builtins.flexitac_vis_count = 0
    builtins.flexitac_vis_last = 0.0

    def refresh_views(event):
        if builtins.flexitac_vis_error is not None or not sim.is_playing():
            return
        now = time.monotonic()
        if now - builtins.flexitac_vis_last < VISUAL_PERIOD_S:
            return
        sensor = getattr(builtins, "flexitac_feet", None)
        if sensor is None or not sensor.is_initialized:
            return
        if omni.usd.get_context().get_stage() != stage:
            return
        active_views = getattr(builtins, "flexitac_views", None)
        if active_views is None or len(active_views) != 2:
            return
        if any(not stage.GetPrimAtPath(view.prim_path).IsValid() for view in active_views):
            return

        try:
            data = sensor.data.tactile_points_w_per_sensor[0]
            for index, view in enumerate(active_views):
                points = data[index]
                colors = (points[:, 3] <= RED_THRESHOLD).to(torch.int64)
                view.set_visibility(True)
                view.visualize(translations=points[:, :3], marker_indices=colors)
            builtins.flexitac_vis_last = now
            builtins.flexitac_vis_count += 1
        except Exception as exc:
            builtins.flexitac_vis_error = repr(exc)

    stream = omni.kit.app.get_app_interface().get_post_update_event_stream()
    builtins.flexitac_vis_subscription = stream.create_subscription_to_pop(refresh_views)


class FlexiTacLiveHeatmap:
    """Show latest two-foot taxel pressures in an Isaac Sim window."""

    def __init__(self, sim, stage):
        self.sim = sim
        self.stage = stage
        self.version = HEATMAP_VERSION
        self.last_update = 0.0
        self.update_count = 0
        self.error = None
        self.providers = [ui.ByteImageProvider(), ui.ByteImageProvider()]
        self.image_frames = []

        self.window = ui.Window("FlexiTac Live Pressure Heatmap", width=700, height=360)
        with self.window.frame:
            with ui.VStack(spacing=8):
                self.scale_label = ui.Label("Estimated pressure (kPa) | Press Play")
                with ui.HStack(spacing=12):
                    self.foot_labels = []
                    for index, name in enumerate(("Left foot", "Right foot")):
                        with ui.VStack(spacing=5):
                            label = ui.Label(f"{name} | waiting for data")
                            self.foot_labels.append(label)
                            frame = ui.Frame(width=256, height=96)
                            provider = self.providers[index]
                            frame.set_build_fn(
                                lambda provider=provider: ui.ImageWithProvider(
                                    provider,
                                    width=256,
                                    height=96,
                                    fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                                )
                            )
                            self.image_frames.append(frame)
                            frame.rebuild()

        stream = omni.kit.app.get_app_interface().get_post_update_event_stream()
        self.subscription = stream.create_subscription_to_pop(self._on_update)

    @staticmethod
    def _rgba_image(grid_kpa, scale_kpa):
        intensity = np.clip(grid_kpa / scale_kpa, 0.0, 1.0)
        rgba = np.empty((*intensity.shape, 4), dtype=np.uint8)
        rgba[..., 0] = (255.0 * np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0.0, 1.0)).astype(np.uint8)
        rgba[..., 1] = (255.0 * np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0.0, 1.0)).astype(np.uint8)
        rgba[..., 2] = (255.0 * np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0.0, 1.0)).astype(np.uint8)
        rgba[..., 3] = 255
        return np.repeat(np.repeat(rgba, 8, axis=0), 8, axis=1)

    def _on_update(self, event):
        if self.error is not None or not self.window.visible or not self.sim.is_playing():
            return
        if omni.usd.get_context().get_stage() != self.stage:
            return
        now = time.monotonic()
        if now - self.last_update < HEATMAP_PERIOD_S:
            return

        sensor = getattr(builtins, "flexitac_feet", None)
        if sensor is None or not sensor.is_initialized:
            return

        try:
            tactile = sensor.data.tactile_points_w_per_sensor
            values = tactile[0, :, :, 3].detach().cpu().numpy()
            forces = values * sensor.cfg.max_force if sensor.cfg.normalize_forces else values
            pressures = forces / (sensor.cfg.point_distance ** 2) / 1000.0
            scale_kpa = max(2.0, float(pressures.max()))

            for index, name in enumerate(("Left foot", "Right foot")):
                grid = pressures[index].reshape(sensor.cfg.num_rows, sensor.cfg.num_cols)
                image = self._rgba_image(grid, scale_kpa)
                self.providers[index].set_bytes_data(
                    bytearray(image.tobytes()), [image.shape[1], image.shape[0]]
                )
                self.image_frames[index].rebuild()
                self.foot_labels[index].text = f"{name} | max {pressures[index].max():.3f} kPa"

            self.scale_label.text = f"Estimated pressure (kPa) | color scale: 0 to {scale_kpa:.2f}"
            self.last_update = now
            self.update_count += 1
            builtins.flexitac_heatmap_count = self.update_count
        except Exception as exc:
            self.error = repr(exc)
            builtins.flexitac_heatmap_error = self.error
            print("FlexiTac heatmap error:", self.error)

    def close(self):
        self.subscription = None
        self.window.visible = False


def ensure_live_heatmap(sim, stage):
    existing = getattr(builtins, "flexitac_heatmap", None)
    if (
        existing is not None
        and getattr(existing, "stage", None) == stage
        and getattr(existing, "version", None) == HEATMAP_VERSION
    ):
        existing.window.visible = True
        return
    if existing is not None:
        existing.close()

    builtins.flexitac_heatmap_error = None
    builtins.flexitac_heatmap_count = 0
    builtins.flexitac_heatmap = FlexiTacLiveHeatmap(sim, stage)


async def setup_flexitac_two_feet():
    if getattr(builtins, "_flexitac_setup_running", False):
        print("FlexiTac setup is already running.")
        return

    builtins._flexitac_setup_running = True
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Open the saved USD scene first.")

        for path in (LEFT_FOOT, RIGHT_FOOT):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing rigid foot prim: {path}")

        ground = stage.GetPrimAtPath(GROUND)
        if not ground.IsValid() or not ground.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Missing ground mesh: {GROUND}")

        physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if physics_scenes:
            physics_scene_path = physics_scenes[0].GetPath().pathString
            print("Using physics scene:", physics_scene_path)
        else:
            physics_scene_path = PHYSICS_SCENE
            UsdPhysics.Scene.Define(stage, physics_scene_path)
            print("Created physics scene:", physics_scene_path)

        existing = getattr(builtins, "flexitac_feet", None)
        if existing is not None and getattr(existing, "stage", None) == stage:
            sim = SimulationContext.instance()
            if sim is not None and sim.get_physics_context() is not None:
                ensure_live_update(sim)
                ensure_live_visualization(sim, stage)
                ensure_live_heatmap(sim, stage)
            print("FlexiTac two-foot sensor already exists on this stage.")
            return

        sim = SimulationContext.instance()
        if sim is None:
            sim = SimulationContext(
                physics_dt=1.0 / 120.0,
                rendering_dt=1.0 / 60.0,
                stage_units_in_meters=1.0,
                physics_prim_path=physics_scene_path,
                set_defaults=False,
                backend="torch",
                device="cuda:0",
                stage=stage,
            )
        if sim.get_physics_context() is None:
            await sim.initialize_simulation_context_async()

        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError("The USD stage changed during setup; run this file again.")

        for path in (LEFT_FOOT, RIGHT_FOOT):
            activate_contact_sensors(path, threshold=0.0)

        cfg = WarpSdfTactileSensorCfg(
            prim_path=ROBOT,
            elastomer_prim_paths=[LEFT_FOOT, RIGHT_FOOT],
            num_rows=12,
            num_cols=32,
            point_distance=0.005,
            normal_axis=2,
            normal_offset=0.0,
            patch_offset_pos_b_per_elastomer=[
                (0.005, -0.051175, -0.030 - LEFT_CALIBRATION_DOWN_M),
                (-0.005, -0.051175, -0.030 - RIGHT_CALIBRATION_DOWN_M),
            ],
            target_mesh_prim_path=GROUND,
            mesh_use_signed_distance=True,
            mesh_signed_distance_method="winding",
            debug_vis=False,
        )

        builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        builtins.flexitac_feet = WarpSdfTactileSensor(cfg)
        ensure_live_update(sim)
        ensure_live_visualization(sim, stage)
        ensure_live_heatmap(sim, stage)
        print("FlexiTac two-foot setup ready. Press Play to initialize and update.")
    finally:
        builtins._flexitac_setup_running = False


run_coroutine(setup_flexitac_two_feet())
