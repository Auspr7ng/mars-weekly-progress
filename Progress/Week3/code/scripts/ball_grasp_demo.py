"""Two independent free balls, continuous scene tactile sensing, live offsets."""
import argparse
import os
from pathlib import Path
from queue import SimpleQueue
import time
import webbrowser

import numpy as np
import warp as wp

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
wp.config.kernel_cache_dir = str(PROJECT / ".cache/warp")

from sunday_a1_mjlab.ball_grasp import BALL_ENTITIES, BALL_RADIUS, CONTACT_SENSORS, GRIP_TARGET, BallGraspBackend
from sunday_a1_mjlab.tactile_display import TactileViewer, save_snapshot, save_foot_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    output = PROJECT.parent / "results/ball_grasp"
    output.mkdir(parents=True, exist_ok=True)
    grasp = BallGraspBackend()
    backend = grasp.runtime
    for _ in range(50):
        result = backend.step([GRIP_TARGET] * 4)
    viewer = TactileViewer(backend.model, backend.data, port=args.port,
                           prefix=backend.prefix, backend="mjlab GPU · dual ball grasp",
                           initial_camera_hand="both", feet=True)
    viewer.left.value = viewer.right.value = GRIP_TARGET * 1000
    viewer.auto.visible = False
    viewer.server.gui.add_markdown(
        "### Hands + feet · eight tactile arrays\n"
        "Two grasped balls + four 32×32 toe/heel arrays in this same simulation.\n"
        "Heatmaps are proximity signals; forces below are actual ball contact forces.", order=-30)
    events = SimpleQueue()
    sliders = {"left": viewer.left, "right": viewer.right}
    spheres, state_labels, offsets = {}, {}, {}
    reset_buttons = {hand: viewer.server.gui.add_button(f"Reset {hand} ball / and grasp", order=-29 + i)
                     for i, hand in enumerate(BALL_ENTITIES)}
    for i, (hand, entity) in enumerate(BALL_ENTITIES.items()):
        release = viewer.server.gui.add_button(f"Release {hand} ball", order=-27 + i)
        release.on_click(lambda _, hand=hand: events.put(("release", hand, None)))
        state_labels[hand] = viewer.server.gui.add_markdown("", order=-25 + i)
        with viewer.server.gui.add_folder(f"{hand.title()} ball position (mm)", order=-23 + i,
                                          expand_by_default=False):
            viewer.server.gui.add_markdown("Offset from fingertip center in hand axes.\n"
                "Moving a slider repositions this ball and recloses this hand.")
            offsets[hand] = [viewer.server.gui.add_slider(
                f"{hand.title()} ball {axis} (mm)", min=-limit, max=limit, step=.5, initial_value=0.)
                for axis, limit in (("X", 8.), ("Y", 6.), ("Z", 20.))]

        def queue_reset(_, hand=hand):
            events.put(("reset", hand, tuple(s.value for s in offsets[hand])))

        reset_buttons[hand].on_click(queue_reset)
        for control in offsets[hand]:
            control.on_update(queue_reset)
        gid = backend.model.geom(entity + "/sphere").id
        handle = viewer.server.scene.add_icosphere(
            "/" + entity, radius=BALL_RADIUS, color=(255, 90, 12) if hand == "left" else (25, 130, 255))
        spheres[hand] = gid, handle
    started, last_draw, count = time.perf_counter(), 0., 50
    url = f"http://127.0.0.1:{viewer.server.get_port()}"
    print(f"Dual ball grasp ready: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while not viewer.stopped and (not args.duration or time.perf_counter() - started < args.duration):
            tick = time.perf_counter()
            pending = {}
            while not events.empty():
                action, hand, offset = events.get()
                pending[hand] = action, offset
            for hand, (action, offset) in pending.items():
                if action == "reset":
                    grasp.reset_hand(hand, offset)
                    sliders[hand].value = GRIP_TARGET * 1000
                else:
                    sliders[hand].value = 30.
                viewer.paused.value = False
            if not viewer.paused.value:
                backend.set_foot_tilts(viewer.left_foot_tilt.value, viewer.right_foot_tilt.value)
                result = backend.step([viewer.left.value / 1000] * 2 + [viewer.right.value / 1000] * 2)
                count += 1
            if tick - last_draw > .1:
                viewer.update(result, backend.data.time, count,
                              foot_result=backend.env.scene["feet_tactile"].data)
                for hand in BALL_ENTITIES:
                    gid, handle = spheres[hand]
                    handle.position = backend.data.geom_xpos[gid]
                    contact = backend.env.scene[CONTACT_SENSORS[hand]].data
                    forces = np.linalg.norm(contact.force[0].detach().cpu().numpy(), axis=-1)
                    found = contact.found[0].detach().cpu().numpy()
                    state_labels[hand].content = (
                        f"**{hand.title()}** · ball height **{backend.data.geom_xpos[gid, 2] * 1000:.1f} mm** · "
                        f"ball contacts **{int((found > 0).sum())}/2** · "
                        f"forces **{forces[0]:.2f}, {forces[1]:.2f} N**")
                last_draw = tick
            if viewer.want_snapshot:
                save_snapshot(result, output, "live_heatmaps")
                save_foot_snapshot(backend.env.scene["feet_tactile"].data, output, "live_foot_heatmaps")
                viewer.want_snapshot = False
            time.sleep(max(0, .02 - (time.perf_counter() - tick)))
    finally:
        viewer.close()
        grasp.close()


if __name__ == "__main__":
    main()
