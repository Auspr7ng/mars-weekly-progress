"""Head-mounted spring joystick for bounded body-frame velocity commands."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

MAX_LENGTH=.30
DEAD_ZONE=.025
JOYSTICK_DEAD_ZONE=.015


def joystick_to_command(vector,held,forward_limit=.08,yaw_limit=.12,backward_limit=0.):
    """Spring-centred body-frame throttle/steering; release always commands zero."""
    if not held:return np.zeros(3),dict(mode='Stopped',requested_speed=0.)
    v=np.asarray(vector,dtype=float)[:2]
    if not np.isfinite(v).all():raise ValueError('Nonfinite joystick input')
    def axis(value,limit,minimum):
        if abs(value)<=JOYSTICK_DEAD_ZONE or limit<=0:return 0.
        amount=float(np.clip((abs(value)-JOYSTICK_DEAD_ZONE)/(MAX_LENGTH-JOYSTICK_DEAD_ZONE),0.,1.))
        floor=min(minimum,limit)
        return math.copysign(floor+(limit-floor)*amount,value)
    forward=axis(v[0],forward_limit if v[0]>=0 else backward_limit,.03 if v[0]>=0 else .04)
    yaw=axis(v[1],yaw_limit,.08)
    mode='Walking' if abs(forward)>.001 else ('Turning' if abs(yaw)>.001 else 'Stopped')
    return np.array([forward,0.,yaw]),dict(mode=mode,requested_speed=abs(forward))


def vector_to_command(vector,yaw):
    """Map a horizontal world arrow to forward speed and steering, never forces."""
    v=np.asarray(vector,dtype=float)[:2]
    if not np.isfinite(v).all() or not math.isfinite(yaw):raise ValueError('Nonfinite arrow input')
    length=float(np.linalg.norm(v))
    if length<=DEAD_ZONE:return np.zeros(3),dict(mode='Stopped',requested_speed=0.,heading_error=0.)
    speed=.03+.05*np.clip((length-DEAD_ZONE)/(MAX_LENGTH-DEAD_ZONE),0.,1.)
    desired=math.atan2(v[1],v[0]);error=math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
    turn=0. if abs(error)<.04 else float(np.clip(.8*error,-.12,.12))
    forward=0. if abs(error)>.60 else float(speed*math.cos(error)**2)
    return np.array([forward,0.,turn]),dict(mode='Turning' if forward==0 else 'Walking',requested_speed=float(speed),heading_error=error)


@dataclass
class ArrowState:
    vector: np.ndarray=field(default_factory=lambda:np.zeros(3))
    gestures: int=0
    updates: int=0
    last_phase: str='idle'
    held: bool=False
    owner: int | None=None

    def set_vector(self,value):
        v=np.asarray(value,dtype=float).copy()
        if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid arrow endpoint')
        v[2]=0.;length=float(np.linalg.norm(v[:2]))
        if length>MAX_LENGTH:v*=MAX_LENGTH/length
        self.vector=v

    def drag(self,phase,value,client_id):
        if phase=='start':self.held=True;self.owner=client_id;self.gestures+=1
        if client_id!=self.owner:return
        self.updates+=1;self.last_phase=phase
        if phase=='end':self.held=False;self.owner=None;self.set_vector(np.zeros(3))
        else:self.set_vector(value)

    def release(self,reason='stop'):
        self.held=False;self.owner=None;self.set_vector(np.zeros(3));self.last_phase=reason


class HeadArrow:
    def __init__(self,server,model,data,status_path,follow=None,limits=None):
        self.server=server;self.model=model;self.lock=threading.RLock();self.state=ArrowState()
        self.status_path=Path(status_path);self.status_path.parent.mkdir(parents=True,exist_ok=True)
        self.head=model.geom('robot/collision_head_1').id;self.root=model.body('robot/base_link').id
        self.last_write=0.;self.last_root=np.array(data.qpos[:3]);self.reference_z=float(data.qpos[2]);self.camera_offsets={};self.follow_was_enabled=True;self.last_command=np.zeros(3)
        self.last_mode=dict(mode='Stopped',requested_speed=0.,heading_error=0.)
        self.limits=limits or dict(forward_limit=.08,yaw_limit=.12,backward_limit=0.)
        self.nonzero_command_steps=0;self.last_nonzero_command=np.zeros(3)
        self.performance={}
        self.follow=follow if follow is not None else server.gui.add_checkbox('Follow camera',initial_value=True)
        self.readout=server.gui.add_markdown('**Stopped** · Drag an axis or the planar handle above the head')
        self.frame=server.scene.add_frame('/walk_command',show_axes=False)
        self.axes=server.scene.add_frame('/walk_command/axes',axes_length=.055,axes_radius=.0025)
        self.arrow=server.scene.add_arrows('/walk_command/direction',np.array([[[0.,0,0],[.001,0,0]]],dtype=np.float32),
                                           colors=(24,185,206),shaft_radius=.004,head_radius=.014,head_length=.035,visible=False)
        self.tip=server.scene.add_icosphere('/walk_command/tip',radius=.012,color=(24,185,206))
        self.handle=server.scene.add_transform_controls('/walk_command/drag',scale=.12,line_width=3.,
                       active_axes=(True,True,False),disable_rotations=True,depth_test=False,
                       translation_limits=((-MAX_LENGTH,MAX_LENGTH),(-MAX_LENGTH,MAX_LENGTH),(0.,0.)))
        self.label=server.scene.add_label('/walk_command/label','Hold to move / release to stop',position=(0,0,.09))
        @self.handle.on_update
        async def dragged(event):
            with self.lock:
                self.state.drag(event.phase,event.target.position,event.client_id)
                self.handle.position=self.state.vector
        self.update(data,None,None)

    def stop(self):
        with self.lock:
            self.state.release()
            self.handle.position=(0.,0.,0.)

    def cache_camera(self,client,data):
        root=np.array([data.qpos[0],data.qpos[1],self.reference_z])
        self.camera_offsets[id(client)]=(np.array(client.camera.position)-root,np.array(client.camera.look_at)-root)

    def command(self,data):
        rotation=data.xmat[self.root].reshape(3,3);yaw=math.atan2(rotation[1,0],rotation[0,0])
        with self.lock:
            clients=self.server.get_clients()
            if self.state.held and self.state.owner not in clients:self.state.release('client_disconnected');self.handle.position=(0.,0.,0.)
            command,mode=joystick_to_command(self.state.vector,self.state.held,**self.limits)
            if np.any(command):self.nonzero_command_steps+=1;self.last_nonzero_command=command.copy()
            self.last_command=command;self.last_mode=mode
            return command.tolist()

    def update(self,data,hands,feet):
        with self.lock:
            vector=self.state.vector.copy();mode=dict(self.last_mode)
            rotation=data.geom_xmat[self.head].reshape(3,3)
            extent=(np.abs(rotation)@self.model.geom_size[self.head])[2]
            self.frame.position=data.geom_xpos[self.head]+np.array([0,0,extent+.10])
            body=data.xmat[self.root].reshape(3,3);yaw=math.atan2(body[1,0],body[0,0])
            self.frame.wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))
            self.arrow.visible=bool(np.any(np.abs(vector[:2])>JOYSTICK_DEAD_ZONE))
            if self.arrow.visible:self.arrow.points=np.array([[np.zeros(3),vector]],dtype=np.float32)
            self.tip.position=vector
            self.arrow.colors=np.array((239,158,42) if mode['mode']=='Turning' else (24,185,206),dtype=np.uint8)
            root=np.array(data.qpos[:3]);self.last_root=root
            if self.follow.value:
                for client in self.server.get_clients().values():
                    if not self.follow_was_enabled:self.cache_camera(client,data)
                    offsets=self.camera_offsets.get(id(client))
                    if offsets is not None:
                        base=np.array([root[0],root[1],self.reference_z])
                        client.camera.position=base+offsets[0]
                        client.camera.look_at=base+offsets[1]
            self.follow_was_enabled=self.follow.value
            velocity=np.zeros(6);mujoco.mj_objectVelocity(self.model,data,mujoco.mjtObj.mjOBJ_XBODY,self.root,velocity,1)
            self.readout.content=f"**{mode['mode']}** · Speed {self.last_command[0]:.3f} / {velocity[3]:.3f} m/s · Turn {self.last_command[2]:+.2f} rad/s"
            now=time.monotonic()
            if now-self.last_write>=1.:
                self.last_write=now
                report=dict(time=time.time(),mode=mode['mode'],joystick_vector_body_m=vector.tolist(),held=self.state.held,
                    requested_speed_m_s=mode['requested_speed'],command=self.last_command.tolist(),
                    actual_forward_m_s=float(velocity[3]),actual_yaw_rate_rad_s=float(velocity[2]),
                    root_xyz_m=root.tolist(),drag_gestures=self.state.gestures,drag_updates=self.state.updates,
                    last_drag_phase=self.state.last_phase,connected_clients=len(self.server.get_clients()),
                    nonzero_command_steps=self.nonzero_command_steps,last_nonzero_command=self.last_nonzero_command.tolist(),joystick_limits=self.limits,
                    hand_distance_max=None if hands is None else float(hands.max()),
                    foot_distance_max=None if feet is None else float(feet.max()),
                    foot_distance_max_per_sensor=None if feet is None else feet.reshape(len(feet),-1).max(axis=1).tolist(),
                    tactile_kind='unchanged Week3 distance response; no force calibration',**self.performance)
                temp=self.status_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');temp.replace(self.status_path)
