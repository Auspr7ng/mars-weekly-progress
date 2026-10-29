"""Render a GPU physics rollout: close/hold, open/drop, reset/hold."""
import os
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

from matplotlib.font_manager import findfont
from sunday_a1_mjlab.ball_grasp import GRIP_TARGET, TASK_ID
from sunday_a1_mjlab.flexitac import FINGERS
from sunday_a1_mjlab.tactile_backend import MjlabTactileBackend
from sunday_a1_mjlab.tactile_display import colorize


def main():
    output = PROJECT.parent / "results/ball_grasp"
    output.mkdir(parents=True, exist_ok=True)
    backend = MjlabTactileBackend(task=TASK_ID)
    model, data = backend.model, backend.data
    model.vis.global_.offwidth, model.vis.global_.offheight = 860, 720
    renderer = mujoco.Renderer(model, height=720, width=860)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.geom_xpos[model.geom("ball/sphere").id] + (0, 0, -.012)
    camera.distance, camera.azimuth, camera.elevation = .24, 210, -15
    font_path = findfont("DejaVu Sans")
    title, label, small = [ImageFont.truetype(font_path, n) for n in (21, 17, 14)]
    writer = imageio_ffmpeg.write_frames(str(output / "ball_grasp.mp4"),
        size=(1280, 720), fps=25, codec="libx264", macro_block_size=1)
    writer.send(None)
    try:
        for stage, targets in (("Grasp and hold", [GRIP_TARGET] * 2 + [.03] * 2),
                               ("Open and release", [.03] * 4),
                               ("Reset and grasp", [GRIP_TARGET] * 2 + [.03] * 2)):
            if stage.startswith("Reset"):
                backend.env.reset()
            for step in range(150):
                result = backend.step(targets)
                if step % 2:
                    continue
                points = result.tactile_points_w_per_sensor[0].detach().cpu().numpy()
                renderer.update_scene(data, camera=camera)
                for xyzr in points[:2].reshape(-1, 4):
                    if xyzr[3] <= .01:
                        continue
                    scene = renderer.scene
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array((.0005,) * 3), xyzr[:3].astype(float), np.eye(3).ravel(),
                        np.array((1, .05, .05, 1), dtype=np.float32))
                    scene.ngeom += 1
                canvas = Image.new("RGB", (1280, 720), "#f7f9fc")
                canvas.paste(Image.fromarray(renderer.render()), (0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 860, 55), fill="#172032")
                draw.text((18, 10), f"Sunday A1 · mjlab GPU · {stage}", font=title, fill="white")
                draw.text((880, 14), "Sphere SDF · four arrays", font=title, fill="#172032")
                for index, finger in enumerate(FINGERS):
                    y = 58 + index * 145
                    draw.text((882, y), finger.name, font=label, fill="#172032")
                    response = points[index, :, 3].reshape(12, 32)
                    panel = colorize(response).repeat(8, 0).repeat(10, 1)
                    canvas.paste(Image.fromarray(panel), (890, y + 25))
                    draw.text((882, y + 124), f"active {(response > .01).sum()}/384", font=small, fill="#172032")
                contact = backend.env.scene["grip_contacts"].data
                forces = np.linalg.norm(contact.force[0].detach().cpu().numpy(), axis=-1)
                draw.text((18, 670), f"Real contact forces: {forces[0]:.2f}, {forces[1]:.2f} N",
                          font=title, fill="white", stroke_width=2, stroke_fill="#172032")
                draw.text((880, 664), "28 mm sphere · 40 g · gravity -9.81 m/s²", font=small, fill="#172032")
                draw.text((880, 690), "Heatmaps are not calibrated pressure", font=small, fill="#172032")
                writer.send(np.asarray(canvas))
                if step == 148:
                    canvas.save(output / ("released.png" if stage.startswith("Open") else "held.png"))
            print(stage, flush=True)
    finally:
        writer.close()
        renderer.close()
        backend.close()


if __name__ == "__main__":
    main()
