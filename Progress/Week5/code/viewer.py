"""Week5 viewer: original distance heatmaps and native gamepad commands."""
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from sunday_locomotion.arrow_control import HeadArrow
from week3_tactile.tactile_display import rotation_quat, hand_heatmap_image, foot_heatmap_image

WEEK = Path(__file__).resolve().parents[1]


class GamepadArrow(HeadArrow):
    def __init__(self, control, *args, **kwargs):
        self.control = control
        super().__init__(*args, **kwargs)
        self.tip.visible = False
        self.label.text = 'Xbox / centre sticks to stop'

    def stop(self):
        self.control.stop()
        super().stop()

    def command(self, data):
        with self.lock:
            if not self.server.get_clients():
                self.control.stop('Open viewer to enable control')
            s = self.control.sample()
            command = np.asarray(s['command'])
            self.last_command = command
            self.last_mode = dict(mode='Walking' if command[0] else ('Turning' if command[2] else 'Stopped'),
                                  requested_speed=abs(command[0]))
            self.state.vector = np.array([command[0]/.12*.25, command[2]/.25*.25, 0.])
            self.state.held = bool(np.any(command))
            if np.any(command):
                self.nonzero_command_steps += 1
                self.last_nonzero_command = command.copy()
            self.performance['gamepad'] = s
            self.performance['input_backend'] = 'pygame_sdl_standard'
            return command.tolist()

    def update(self, data, hands, feet):
        super().update(data, hands, feet)
        s = self.control.sample()
        self.readout.content = (f"**{s['reason']}**\n\n{s['device'] or 'No controller'} · "
                                f"Left Y {s['y']:+.3f} / Left X {s['x']:+.3f}\n\n"
                                f"Command {self.last_command[0]:+.3f} m/s · {self.last_command[2]:+.3f} rad/s")


class Viewer:
    def __init__(self, model, data, control, reader, port):
        import viser
        self.model = model
        self.server = viser.ViserServer(host='127.0.0.1', port=port)
        self.server.scene.set_up_direction('+z')
        self.server.gui.configure_theme(show_share_button=False)
        self.reset_requested = self.stopped = self.save_requested = False
        self.last_trace = 0.
        self.trace = (WEEK / 'results/simulation_trace.jsonl').open('a', encoding='utf-8', buffering=1)
        self.server.gui.add_markdown('## Sunday A1 · Xbox Controller\nWeek5 · Native Bluetooth/USB input')
        self.server.gui.add_markdown(('**Hold LB to move.** ' if control.require_lb else '**Left stick controls walking.** ')+ 'Up/down: forward/reverse. Left/right: steer. Diagonal: walk and turn. Right stick reserved for arms. '
                                     '**B: stop. Menu: reset** (release LB first).\n\n'
                                     'Centre left stick to stop/re-arm. Commands follow robot heading, independent of the camera. Camera: drag empty space to orbit, right-drag to pan, scroll to zoom.')
        self.paused = self.server.gui.add_checkbox('Pause simulation', initial_value=False)
        follow = self.server.gui.add_checkbox('Follow robot position', initial_value=True)
        self.arrow = GamepadArrow(control, self.server, model, data, WEEK/'results/live_status.json', follow=follow, limits=control.limits)
        @self.paused.on_update
        def pause(_):
            self.arrow.stop()
        @self.server.gui.add_button('Stop walking (B)').on_click
        def stop(_):
            self.arrow.stop()
        @self.server.gui.add_button('Reset episode (Menu)').on_click
        def reset(_):
            self.reset_requested = True
        reader.on_reset = lambda: setattr(self, 'reset_requested', True)
        @self.server.gui.add_button('Save heatmaps').on_click
        def save(_):
            self.save_requested = True
        @self.server.gui.add_button('Close viewer').on_click
        def close(_):
            self.arrow.stop()
            self.stopped = True
        self.status = self.server.gui.add_markdown('Starting native MuJoCo physics...')
        self.feet_images, self.hand_images = [], []
        with self.server.gui.add_folder('Feet · original distance response', expand_by_default=False):
            for i in range(4):
                self.feet_images.append(self.server.gui.add_image(foot_heatmap_image(np.zeros((32,32)), i), label=['Left toe','Left heel','Right toe','Right heel'][i]))
        with self.server.gui.add_folder('Hands · original distance response', expand_by_default=False):
            for i in range(2):
                self.hand_images.append(self.server.gui.add_image(hand_heatmap_image(np.zeros((4,12,32)), i), label=['Left hand','Right hand'][i]))
        self.meshes = []
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[gid] != 1:
                continue
            mid = model.geom_dataid[gid]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            mesh = self.server.scene.add_mesh_simple(f'/robot/{gid}', vertices=np.array(model.mesh_vert[va:va+vn]),
                                                      faces=np.array(model.mesh_face[fa:fa+fn]), color=(185,194,204), side='double')
            self.meshes.append((gid, mesh))
        self.server.scene.add_grid('/ground', width=20, height=20, plane='xy', cell_size=.1, section_size=1.)
        @self.server.on_client_connect
        def camera(client):
            center = data.qpos[:3]+np.array([0,0,.05])
            client.camera.up_direction = (0,0,1)
            client.camera.position = center+np.array([.60,-.80,.60])
            client.camera.look_at = center
            self.arrow.cache_camera(client, data)
            @client.camera.on_update
            def moved(_):
                self.arrow.cache_camera(client, data)

    def update(self, data, hands, feet, label):
        hand, foot = hands.response[0].cpu().numpy(), feet.response[0].cpu().numpy()
        scale = max(.005, float(foot.max()))
        with self.server.atomic():
            for gid, mesh in self.meshes:
                mesh.position = data.geom_xpos[gid]
                mesh.wxyz = rotation_quat(data.geom_xmat[gid])
            for i, image in enumerate(self.feet_images):
                image.image = foot_heatmap_image(foot[i], i, scale)
            for i, image in enumerate(self.hand_images):
                image.image = hand_heatmap_image(hand, i)
            self.arrow.update(data, hand, foot)
            self.status.content = label
        now = time.monotonic()
        if now-self.last_trace >= .1:
            self.last_trace = now
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_XBODY, self.arrow.root, velocity, 1)
            self.trace.write(json.dumps(dict(time=time.time(), simulation_time_s=float(data.time),
                                  input=self.arrow.control.sample(), command=self.arrow.last_command.tolist(),
                                  root_xyz_m=data.qpos[:3].tolist(), actual_forward_m_s=float(velocity[3]),
                                  actual_yaw_rad_s=float(velocity[2]), status=label))+'\n')
        if self.save_requested:
            self.save_requested = False
            destination = WEEK/'results'/time.strftime('capture_%Y%m%d_%H%M%S')
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination/'distance_arrays.npz', hands=hand, feet=foot, root_qpos=data.qpos.copy())
            self.status.content = f'Saved {destination.name}'

    def close(self):
        self.arrow.stop()
        self.server.stop()
        self.trace.close()
