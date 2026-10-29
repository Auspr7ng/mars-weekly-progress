"""Export an actual mjlab GPU rollout with model, taxels and four heatmaps."""
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
from sunday_a1_mjlab.flexitac import FINGERS
from sunday_a1_mjlab.tactile_backend import MjlabTactileBackend
from sunday_a1_mjlab.tactile_display import colorize


def main():
    output = PROJECT.parent / "results/flexitac"
    output.mkdir(exist_ok=True)
    backend = MjlabTactileBackend()
    model, data = backend.model, backend.data
    model.vis.global_.offwidth, model.vis.global_.offheight = 860, 720
    renderer = mujoco.Renderer(model, height=720, width=860)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.xpos[model.body("robot/base_link").id] + (0., 0., -0.025)
    camera.distance, camera.azimuth, camera.elevation = 1.25, 120, -12
    font_path = findfont("DejaVu Sans")
    title_font, label_font, stats_font = [ImageFont.truetype(font_path, n) for n in (21, 17, 14)]
    writer = imageio_ffmpeg.write_frames(str(output / "four_flexitac_demo.mp4"),
                                         size=(1280, 720), fps=25, codec="libx264",
                                         macro_block_size=1)
    writer.send(None)
    frame_count = 0
    try:
        for name, targets in (
            ("Both open", [0.03] * 4), ("Both closed", [0.] * 4),
            ("Left closed / right open", [0., 0., 0.03, 0.03]),
            ("Left open / right closed", [0.03, 0.03, 0., 0.]),
        ):
            for step in range(150):
                result = backend.step(targets)
                if step % 2:
                    continue
                values = result.tactile_points_w_per_sensor[0].detach().cpu().numpy()
                renderer.update_scene(data, camera=camera)
                for xyzr in values.reshape(-1, 4):
                    scene = renderer.scene
                    if scene.ngeom >= scene.maxgeom:
                        raise RuntimeError("Insufficient render geometry capacity")
                    value = float(xyzr[3])
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array((0.00065,) * 3), xyzr[:3].astype(np.float64),
                                       np.eye(3).ravel(), np.array((value, 0.15, 1 - value, 1), dtype=np.float32))
                    scene.ngeom += 1
                canvas = Image.new("RGB", (1280, 720), "#f7f9fc")
                canvas.paste(Image.fromarray(renderer.render()), (0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 860, 55), fill="#172032")
                draw.text((18, 10), f"Sunday A1 · mjlab GPU · {name}", font=title_font, fill="white")
                draw.text((880, 14), "Four FlexiTac · 12 × 32", font=title_font, fill="#172032")
                for index, finger in enumerate(FINGERS):
                    y = 58 + index * 151
                    draw.text((882, y), finger.name, font=label_font, fill="#172032")
                    response = values[index, :, 3].reshape(12, 32)
                    panel = colorize(response).repeat(8, 0).repeat(10, 1)
                    panel[::8] = (panel[::8] * 0.65).astype(np.uint8)
                    panel[:, ::10] = (panel[:, ::10] * 0.65).astype(np.uint8)
                    canvas.paste(Image.fromarray(panel), (890, y + 25))
                    draw.text((882, y + 124), f"max {response.max():.3f} · mean {response.mean():.3f} · "
                              f"active {(response > 0.01).sum()}/384", font=stats_font, fill="#172032")
                draw.text((880, 677), "Blue 0 → red 1 · normalized SDF response", font=stats_font, fill="#172032")
                draw.text((880, 699), "Not calibrated pressure", font=stats_font, fill="#172032")
                writer.send(np.asarray(canvas))
                if name == "Both closed" and step == 148:
                    canvas.save(output / "four_flexitac_demo_closed.png")
                frame_count += 1
            print(f"Recorded {name}", flush=True)
    finally:
        writer.close()
        renderer.close()
        backend.close()
    print(f"Saved {frame_count} frames, 12 seconds: {output / 'four_flexitac_demo.mp4'}")


if __name__ == "__main__":
    main()
