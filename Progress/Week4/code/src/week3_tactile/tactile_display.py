"""Heatmap exports and a single local Viser window for robot + four arrays."""
import json
import os
from pathlib import Path
from functools import lru_cache

import mujoco
import numpy as np

from .flexitac import FINGERS, ROWS, COLS

PROJECT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))


def colorize(response):
    value = np.clip(response, 0, 1)
    return (np.stack((value, np.full_like(value, 0.15), 1 - value), axis=-1) * 255).astype(np.uint8)


def physical_heatmap(response, sensor_index):
    """Unfolded finger surface: top=local +Z/root, right=local +X.

    Storage is [X, Z*inward], not image [vertical, horizontal]. Facing
    arrays reverse their stored Z order; correct that before displaying.
    Raw sensor tensors and physical poses are never changed here.
    """
    values = np.asarray(response).reshape(ROWS, COLS).T
    if FINGERS[sensor_index].inward > 0:
        values = values[::-1, :]
    return np.ascontiguousarray(values)


def heatmap_image(response, sensor_index=0, cell_size=6):
    image = colorize(physical_heatmap(response, sensor_index)).repeat(cell_size, 0).repeat(cell_size, 1)
    image[::cell_size] = (image[::cell_size].astype(np.float32) * 0.65).astype(np.uint8)
    image[:, ::cell_size] = (image[:, ::cell_size].astype(np.float32) * 0.65).astype(np.uint8)
    return image


@lru_cache(maxsize=1)
def _heatmap_font():
    import matplotlib
    from PIL import ImageFont
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 13)


def hand_heatmap_image(responses, hand_index):
    """Side-by-side vertical pads with shared, explicitly labelled axes."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (320, 326), (248, 250, 253))
    draw = ImageDraw.Draw(canvas)
    text_font = _heatmap_font()
    ink = (35, 45, 60)
    start = hand_index * 2
    for index, x in zip((start, start + 1), (32, 192)):
        canvas.paste(Image.fromarray(heatmap_image(responses[index], index, cell_size=8)), (x, 42))
        draw.text((x + 48, 10), FINGERS[index].name, font=text_font, fill=ink, anchor="mm")
    draw.text((160, 30), "ROOT (+Z)", font=text_font, fill=ink, anchor="mm")
    draw.text((160, 311), "TIP (-Z)    |    width +X >", font=text_font, fill=ink, anchor="mm")
    return np.asarray(canvas)


def foot_heatmap_image(response, patch_index, scale=1.):
    """32x32 top-down sole view: toe (-Y) at top, +X to the right."""
    from PIL import Image, ImageDraw
    from .feet_tactile import FOOT_PATCHES
    canvas = Image.new("RGB", (288, 310), (248, 250, 253))
    draw = ImageDraw.Draw(canvas)
    values = np.asarray(response).reshape(32, 32)
    image = colorize(values / max(scale, 1.e-6)).repeat(8, 0).repeat(8, 1)
    image[::8] = (image[::8].astype(float) * .65).astype(np.uint8)
    image[:, ::8] = (image[:, ::8].astype(float) * .65).astype(np.uint8)
    canvas.paste(Image.fromarray(image), (16, 30))
    draw.text((144, 12), FOOT_PATCHES[patch_index].name + " | toe (-Y) ^",
              font=_heatmap_font(), fill=(35, 45, 60), anchor="mm")
    draw.text((144, 298), "heel (+Y) v    |    width +X >", font=_heatmap_font(),
              fill=(35, 45, 60), anchor="mm")
    return np.asarray(canvas)


def save_foot_snapshot(result, directory, name):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from .feet_tactile import FOOT_PATCHES
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points = result.tactile_points_w_per_sensor.detach().cpu().numpy().copy()
    distance = result.distance_m.detach().cpu().numpy().copy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("flexitac", [(0, .15, 1), (1, .15, 0)])
    scale = max(.005, float(points[..., 3].max()))
    for index, (patch, ax) in enumerate(zip(FOOT_PATCHES, axes.flat)):
        image = ax.imshow(points[0, index, :, 3].reshape(32, 32), vmin=0, vmax=scale,
                           cmap=cmap, interpolation="nearest", origin="upper",
                           extent=(-40, 40, 40, -40))
        ax.set_title(patch.name)
        ax.set_xlabel("Local X relative to patch center (mm)")
        ax.set_ylabel("Toward toe (-Y) ↑ | heel (+Y) ↓")
    fig.suptitle("Four foot arrays | 32 x 32 each | 2.5 mm pitch")
    fig.colorbar(image, ax=list(axes.flat), label="Normalized proximity response (not calibrated pressure)", shrink=.8)
    fig.savefig(directory / f"{name}.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(directory / f"{name}.npz", tactile_points_w_per_sensor=points,
                        distance_m=distance, heatmaps=points[..., 3].reshape(-1, 4, 32, 32),
                        sensor_names=np.array([p.name for p in FOOT_PATCHES]))
    (directory / f"{name}.json").write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")


def save_snapshot(result, directory, name):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points = result.tactile_points_w_per_sensor.detach().cpu().numpy().copy()
    distance = result.distance_m.detach().cpu().numpy().copy()
    summary = result.summary()
    fig, axes = plt.subplots(1, 4, figsize=(11, 7), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("flexitac", [(0, 0.15, 1), (1, 0.15, 0)])
    for index, (finger, ax) in enumerate(zip(FINGERS, axes.flat)):
        values = physical_heatmap(points[0, index, :, 3], index)
        image = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, interpolation="nearest", aspect="equal",
                          extent=(-12, 12, -32, 32), origin="upper")
        stats = summary[finger.name]
        ax.set_title(f"{finger.name}\nmax {stats['max']:.3f} | mean {stats['mean']:.3f}\n"
                     f"active {stats['active']}/384", fontsize=10)
        ax.set_xlabel("Local X (mm) →")
        ax.set_ylabel("Tip ← local Z (mm) → root")
        ax.set_xticks(np.arange(-12, 13, 2), minor=True)
        ax.set_yticks(np.arange(-32, 33, 2), minor=True)
        ax.grid(which="minor", color="white", alpha=0.2, linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle("Sunday A1 | Vertical tactile surfaces | top = root (+Z), bottom = tip (-Z)")
    fig.colorbar(image, ax=list(axes.flat), label="Normalized proximity response (not calibrated pressure)", shrink=0.8)
    fig.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    np.savez_compressed(directory / f"{name}.npz", tactile_points_w_per_sensor=points,
                        distance_m=distance, sensor_names=np.array([f.name for f in FINGERS]),
                        display_heatmaps=np.stack([physical_heatmap(points[0, i, :, 3], i) for i in range(4)]),
                        display_axes=np.array(["top=local +Z/root", "right=local +X"]))
    (directory / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def rotation_quat(rotation):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).reshape(9))
    return quat


class TactileViewer:
    def __init__(self, model, data, port=8081, prefix="", backend="mjlab", initial_camera_hand=None,
                 feet=False):
        import viser
        self.server = viser.ViserServer(host="127.0.0.1", port=port)
        self.server.scene.set_up_direction("+z")
        self.model, self.data = model, data
        self.prefix = prefix
        self.server.gui.configure_theme(control_width="large", show_share_button=False)
        self.server.gui.main_panel.dock_right()
        self.has_feet = feet
        title = "Sunday A1 · Eight tactile arrays" if feet else "Sunday A1 · Four FlexiTac"
        dimensions = "Hands: 4 × 12×32 · feet: 4 × 32×32" if feet else "12 × 32 per finger · 2 mm pitch"
        self.server.gui.add_markdown(f"## {title}\n{dimensions} · {backend}")
        self.left = self.server.gui.add_slider("Left gripper (mm)", min=0., max=30., step=0.5, initial_value=30.)
        self.right = self.server.gui.add_slider("Right gripper (mm)", min=0., max=30., step=0.5, initial_value=30.)
        self.paused = self.server.gui.add_checkbox("Pause simulation", initial_value=False)
        self.auto = self.server.gui.add_checkbox("Automatic open / close (3 s each)", initial_value=False)
        self.want_snapshot = False
        self.stopped = False
        self.show_taxels = self.server.gui.add_checkbox("Show taxels", initial_value=True)
        self.show_foot_taxels = self.server.gui.add_checkbox("Show foot taxels", initial_value=True) if feet else None
        self.show_ground = self.server.gui.add_checkbox("Show ground", initial_value=True) if feet else None
        self.status = self.server.gui.add_markdown("Initializing live sensors...")
        open_button = self.server.gui.add_button("Open both")
        close_button = self.server.gui.add_button("Close both")
        save_button = self.server.gui.add_button("Save heatmaps + readings")
        stop_button = self.server.gui.add_button("Stop viewer")

        @stop_button.on_click
        def stop(_):
            self.stopped = True

        @open_button.on_click
        def open_both(_):
            self.auto.value = False
            self.left.value = self.right.value = 30.

        @close_button.on_click
        def close_both(_):
            self.auto.value = False
            self.left.value = self.right.value = 0.

        @save_button.on_click
        def snapshot(_):
            self.want_snapshot = True

        for slider in (self.left, self.right):
            @slider.on_update
            def manual(event):
                if event.client is not None:
                    self.auto.value = False

        self.images, self.labels = [], []
        with self.server.gui.add_folder("Live heatmaps · common scale 0 → 1"):
            self.server.gui.add_markdown("**Vertical finger surfaces** · top = root · bottom = tip.\n"
                "Paired maps share local X/Z axes; facing sensors have their stored Z order corrected.")
            for hand_index, hand in enumerate(("Left", "Right")):
                self.images.append(self.server.gui.add_image(
                    hand_heatmap_image(np.zeros((4, ROWS, COLS)), hand_index),
                    label=f"{hand} hand · paired vertical pads", format="png"))
                for _ in range(2):
                    self.labels.append(self.server.gui.add_markdown("Waiting for sample"))
        self.server.gui.add_markdown("Response = normalized distance-shell signal, not calibrated pressure.\n"
                                     "Blue = 0; red = 1. Pause holds the most recent sample.")
        self.foot_images, self.foot_labels = [], []
        self.foot_patch_handles = []
        if feet:
            from .feet_tactile import FOOT_PATCHES, FOOT_SIDE_M
            with self.server.gui.add_folder("Foot heatmaps · four 32 x 32 arrays"):
                self.server.gui.add_markdown("Each foot: **front + heel**, 80×80 mm each, **16 mm center gap**.\n"
                    "Top-down views: top = toe (-Y), right = +X. Shared automatic foot color scale.")
                self.foot_scale_label = self.server.gui.add_markdown("Waiting for foot scale")
                self.left_foot_tilt = self.server.gui.add_slider("Left sole tilt (deg)", min=-5., max=5., step=.5, initial_value=0.)
                self.right_foot_tilt = self.server.gui.add_slider("Right sole tilt (deg)", min=-5., max=5., step=.5, initial_value=0.)
                flatten = self.server.gui.add_button("Flatten feet")
                @flatten.on_click
                def flatten_feet(_):
                    self.left_foot_tilt.value = self.right_foot_tilt.value = 0.
                for index, patch in enumerate(FOOT_PATCHES):
                    self.foot_images.append(self.server.gui.add_image(
                        foot_heatmap_image(np.zeros((32, 32)), index), label=patch.name, format="png"))
                    self.foot_labels.append(self.server.gui.add_markdown("Waiting for foot sample"))
            self.foot_patch_handles = [self.server.scene.add_box(
                f"/foot_patches/{p.name}", color=tuple(int(v * 255) for v in p.color),
                dimensions=(FOOT_SIDE_M, FOOT_SIDE_M, .0003), opacity=.3) for p in FOOT_PATCHES]
            self.foot_points = self.server.scene.add_point_cloud(
                "/foot_taxels", points=np.zeros((4096, 3)), colors=np.zeros((4096, 3), dtype=np.uint8),
                point_size=.0015, point_shape="circle", precision="float32")
        self.mesh_handles = []
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[gid] != 1:
                continue
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            handle = self.server.scene.add_mesh_simple(
                f"/robot/geom_{gid}", vertices=np.array(model.mesh_vert[va:va + vn]),
                faces=np.array(model.mesh_face[fa:fa + fn]), color=(185, 194, 204), side="double")
            self.mesh_handles.append((gid, handle))
        self.patch_handles = [self.server.scene.add_box(
            f"/patches/{f.name}", color=tuple(int(v * 255) for v in f.color),
            dimensions=(0.024, 0.0003, 0.064), opacity=0.35) for f in FINGERS]
        self.points = self.server.scene.add_point_cloud(
            "/taxels", points=np.zeros((4 * ROWS * COLS, 3)),
            colors=np.zeros((4 * ROWS * COLS, 3), dtype=np.uint8),
            point_size=0.0014, point_shape="circle", precision="float32")
        floor_id = int(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
        floor_z = data.geom_xpos[floor_id, 2]
        self.floor = self.server.scene.add_box("/floor", dimensions=(2., 2., 0.002),
                                              color=(42, 48, 60), position=(0, 0, floor_z - 0.002))

        def camera(client, hand=None):
            if hand is None:
                center = data.xpos[model.body(prefix + "base_link").id] + np.array((0, 0, -0.03))
                client.camera.position = center + np.array((0.8, -1.05, 0.48))
                client.camera.look_at = center
            else:
                if hand in ("feet", "soles"):
                    from .feet_tactile import FOOT_PATCHES
                    pair = FOOT_PATCHES
                else:
                    pair = FINGERS if hand == "both" else (FINGERS[:2] if hand == "left" else FINGERS[2:])
                centers = []
                for finger in pair:
                    bid = model.body(prefix + finger.link).id
                    centers.append(data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.asarray(finger.center))
                target = np.mean(centers, axis=0)
                offset = (0., -.28, -.24) if hand == "soles" else (
                    (0., -.34, .30) if hand == "feet" else (
                    (-0.35, -0.32, 0.20) if hand == "both" else (-0.18, -0.11, 0.08))
                )
                client.camera.position = target + np.array(offset)
                client.camera.look_at = target
                if hand == "soles":
                    self.show_ground.value = False
            client.camera.up_direction = (0, 0, 1)

        @self.server.on_client_connect
        def on_connect(client):
            first_update = [True]

            @client.camera.on_update
            def frame_first_view(_):
                if first_update[0]:
                    first_update[0] = False
                    camera(client, initial_camera_hand)

            camera(client, initial_camera_hand)
        cameras = [("Full robot", None), ("Both hands", "both"),
                   ("Left fingers close-up", "left"), ("Right fingers close-up", "right")]
        if feet:
            cameras.append(("Feet close-up", "feet"))
            cameras.append(("Sole sensors underside (hide ground)", "soles"))
        for label, hand in cameras:
            button = self.server.gui.add_button(label)
            button.on_click(lambda event, hand=hand: camera(event.client, hand) if event.client else None)

    def update(self, result, simulation_time, frame, foot_result=None):
        points = result.tactile_points_w_per_sensor[0].detach().cpu().numpy()
        summary = result.summary()
        with self.server.atomic():
            for gid, handle in self.mesh_handles:
                handle.position = self.data.geom_xpos[gid]
                handle.wxyz = rotation_quat(self.data.geom_xmat[gid])
            for finger, handle in zip(FINGERS, self.patch_handles):
                bid = self.model.body(self.prefix + finger.link).id
                rotation = self.data.xmat[bid].reshape(3, 3)
                handle.position = self.data.xpos[bid] + rotation @ np.asarray(finger.center)
                handle.wxyz = rotation_quat(rotation)
            self.points.points = points[..., :3].reshape(-1, 3)
            self.points.colors = colorize(points[..., 3]).reshape(-1, 3)
            self.points.visible = self.show_taxels.value
            for hand_index in range(2):
                self.images[hand_index].image = hand_heatmap_image(points[..., 3], hand_index)
            for index, finger in enumerate(FINGERS):
                stats = summary[finger.name]
                self.labels[index].content = (f"**{finger.name}** · max **{stats['max']:.3f}** · mean **{stats['mean']:.3f}** · "
                                               f"active **{stats['active']}/384**")
            if self.has_feet and foot_result is not None:
                from .feet_tactile import FOOT_PATCHES
                foot_points = foot_result.tactile_points_w_per_sensor[0].detach().cpu().numpy()
                foot_stats = foot_result.summary()
                self.floor.visible = self.show_ground.value
                foot_scale = max(.005, float(foot_points[..., 3].max()))
                self.foot_scale_label.content = (
                    f"Four-foot-map color range: **0 → {foot_scale:.4f}**, normalized contact proxy (auto).")
                self.foot_points.points = foot_points[..., :3].reshape(-1, 3)
                self.foot_points.colors = colorize(foot_points[..., 3]).reshape(-1, 3)
                self.foot_points.visible = self.show_foot_taxels.value
                for index, patch in enumerate(FOOT_PATCHES):
                    bid = self.model.body(self.prefix + patch.link).id
                    rotation = self.data.xmat[bid].reshape(3, 3)
                    self.foot_patch_handles[index].position = self.data.xpos[bid] + rotation @ patch.center
                    self.foot_patch_handles[index].wxyz = rotation_quat(rotation)
                    self.foot_images[index].image = foot_heatmap_image(foot_points[index, :, 3], index, foot_scale)
                    stats = foot_stats[patch.name]
                    self.foot_labels[index].content = (
                        f"max **{stats['max']:.4f}** · mean **{stats['mean']:.4f}** · contact **{stats['active']}/1024**")
            self.status.content = f"Time **{simulation_time:.2f} s** · sensor samples **{frame}** · " + (
                "**PAUSED**" if self.paused.value else "**RUNNING**")

    def close(self):
        self.server.stop()
