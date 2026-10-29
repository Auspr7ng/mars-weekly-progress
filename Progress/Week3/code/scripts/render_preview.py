"""Render the imported model for visual inspection using MuJoCo."""
from pathlib import Path
import struct
import zlib

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
model = mujoco.MjModel.from_xml_path(str(PROJECT / "src/sunday_a1_mjlab/assets/scene.xml"))
model.vis.global_.offwidth = 1000
model.vis.global_.offheight = 800
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
camera = mujoco.MjvCamera()
camera.lookat[:] = (0.0, 0.0, -0.1)
camera.distance = 1.7
camera.azimuth = 120
camera.elevation = -15
with mujoco.Renderer(model, height=800, width=1000) as renderer:
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()


def chunk(kind, payload):
    return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", zlib.crc32(kind + payload))


raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
output = PROJECT.parent / "results/robot_preview.png"
output.parent.mkdir(exist_ok=True)
output.write_bytes(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack("!2I5B", pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
)
print(output)
