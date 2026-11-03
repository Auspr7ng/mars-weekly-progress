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
INTERACTION_UI_VERSION='body_follow_v3'


def drag_to_body_vector(start,end,camera_position,camera_look_at,yaw):
    """Unconstrained pointer-plane motion mapped to the horizontal body plane.

    Screen-up follows the ground away from the camera. The elevation gain is
    bounded so orbiting down to eye level never makes the joystick singular.
    The caller supplies the current camera and current robot heading.
    """
    delta=np.asarray(end,dtype=float)-np.asarray(start,dtype=float)
    view=np.asarray(camera_look_at,dtype=float)-np.asarray(camera_position,dtype=float)
    view/=max(float(np.linalg.norm(view)),1e-9)
    forward=np.array([view[0],view[1],0.])
    if np.linalg.norm(forward)<1e-5:forward=np.array([math.cos(yaw),math.sin(yaw),0.])
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,np.array([0.,0.,1.]))
    up=np.cross(right,view);up/=max(float(np.linalg.norm(up)),1e-9)
    elevation=math.copysign(max(abs(float(view[2])),.35),-float(view[2]) if view[2] else 1.)
    world=right*np.dot(delta,right)+forward*np.dot(delta,up)/elevation
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],0.])


def pointer_delta_to_body_vector(previous,current,camera,anchor,yaw):
    """Map only new mouse motion using the live body frame.

    Deflection is retained in body coordinates when the pointer is stationary.
    Head translation, turning and camera following therefore cannot manufacture
    mouse input. New pointer motion uses the latest camera and robot heading.
    """
    delta=np.asarray(current,dtype=float)-np.asarray(previous,dtype=float)
    if delta.shape!=(2,) or not np.isfinite(delta).all():raise ValueError('Invalid pointer delta')
    if not np.any(delta):return np.zeros(3)
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,np.asarray(camera.wxyz,dtype=float))
    rotation=rotation.reshape(3,3)
    ray=np.asarray(anchor,dtype=float)-camera.position
    depth=max(.05,float(np.dot(ray,rotation[:,2])))
    height=2.*depth*math.tan(float(camera.fov)/2.)
    world_delta=height*(float(camera.aspect)*delta[0]*rotation[:,0]+delta[1]*rotation[:,1])
    # Intersect through the CURRENT handle, rather than the optical axis. The
    # latter gives the wrong direction when the robot is away from screen centre.
    next_ray=ray+world_delta
    distance=max(float(np.linalg.norm(ray)),.05)
    if abs(ray[2])>.20*distance and abs(next_ray[2])>.05*distance and ray[2]*next_ray[2]>0:
        world=next_ray*(ray[2]/next_ray[2])-ray
    else:
        denominator=math.copysign(max(abs(float(ray[2])),.35*distance),float(ray[2]) if ray[2] else -1.)
        world=world_delta-ray*(world_delta[2]/denominator)
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],0.])


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
        self.last_write=0.;self.last_root=np.array(data.qpos[:3]);self.reference_z=float(data.qpos[2]);self.camera_offsets={};self.follow_was_enabled=False;self.last_command=np.zeros(3)
        self.pointer_screen=None;self.frame_yaw=0.;self.camera_updates=0
        self.max_anchor_error=0.;self.max_heading_error=0.;self.held_pose_samples=0
        self.held_yaw_min=None;self.held_yaw_max=None
        self.last_mode=dict(mode='Stopped',requested_speed=0.,heading_error=0.)
        self.limits=limits or dict(forward_limit=.08,yaw_limit=.12,backward_limit=0.)
        self.nonzero_command_steps=0;self.last_nonzero_command=np.zeros(3)
        self.performance={}
        self.follow=follow if follow is not None else server.gui.add_checkbox('Follow robot position',initial_value=False)
        self.readout=server.gui.add_markdown('**Stopped** · Hold and freely drag the turquoise ball above the head')
        self.frame=server.scene.add_frame('/walk_command',show_axes=False)
        self.axes=server.scene.add_frame('/walk_command/axes',axes_length=.055,axes_radius=.0025)
        self.forward=server.scene.add_arrows('/walk_command/forward',np.array([[[0.,0,0],[.09,0,0]]],dtype=np.float32),
                                           colors=(220,65,65),shaft_radius=.0025,head_radius=.008,head_length=.018)
        self.arrow=server.scene.add_arrows('/walk_command/direction',np.array([[[0.,0,0],[.001,0,0]]],dtype=np.float32),
                                           colors=(24,185,206),shaft_radius=.004,head_radius=.014,head_length=.035,visible=False)
        self.tip=server.scene.add_icosphere('/walk_command/tip',radius=.026,color=(24,185,206))
        self.handle=self.tip
        self.label=server.scene.add_label('/walk_command/label','Robot-relative joystick / release to stop',position=(0,0,.09))
        @self.tip.on_drag
        async def dragged(event):
            with self.lock:
                if event.phase=='start':
                    if self.state.held and self.state.owner!=event.client_id:return
                    self.pointer_screen=np.asarray(event.start_screen_pos,dtype=float)
                elif event.client_id!=self.state.owner:return
                if self.pointer_screen is None:return
                self.sync_frame(data)
                c,s=math.cos(self.frame_yaw),math.sin(self.frame_yaw)
                tip_world=np.asarray(self.frame.position)+np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@self.state.vector
                increment=pointer_delta_to_body_vector(self.pointer_screen,event.end_screen_pos,event.client.camera,tip_world,self.frame_yaw) if event.phase!='end' else np.zeros(3)
                vector=self.state.vector+increment
                self.state.drag(event.phase,vector,event.client_id)
                self.pointer_screen=np.asarray(event.end_screen_pos,dtype=float)
                self.tip.position=self.state.vector
                if event.phase=='end' and not self.state.held:self.pointer_screen=None
        self.update(data,None,None)

    def stop(self):
        with self.lock:
            self.state.release()
            self.handle.position=(0.,0.,0.)
            self.pointer_screen=None

    def sync_frame(self,data):
        """The visible basis always follows the current head and body yaw."""
        rotation=data.geom_xmat[self.head].reshape(3,3)
        extent=(np.abs(rotation)@self.model.geom_size[self.head])[2]
        self.frame.position=data.geom_xpos[self.head]+np.array([0,0,extent+.10])
        body=data.xmat[self.root].reshape(3,3)
        self.frame_yaw=math.atan2(body[1,0],body[0,0])
        self.frame.wxyz=(math.cos(self.frame_yaw/2),0.,0.,math.sin(self.frame_yaw/2))

    def cache_camera(self,client,data):
        with self.lock:
            root=np.array([data.qpos[0],data.qpos[1],self.reference_z])
            self.camera_offsets[id(client)]=(np.array(client.camera.position)-root,np.array(client.camera.look_at)-root)
            self.camera_updates+=1

    def command(self,data):
        rotation=data.xmat[self.root].reshape(3,3);yaw=math.atan2(rotation[1,0],rotation[0,0])
        with self.lock:
            clients=self.server.get_clients()
            if self.state.held and self.state.owner not in clients:
                self.state.release('client_disconnected');self.handle.position=(0.,0.,0.);self.pointer_screen=None
            command,mode=joystick_to_command(self.state.vector,self.state.held,**self.limits)
            if np.any(command):self.nonzero_command_steps+=1;self.last_nonzero_command=command.copy()
            self.last_command=command;self.last_mode=mode
            return command.tolist()

    def update(self,data,hands,feet):
        with self.lock:
            vector=self.state.vector.copy();mode=dict(self.last_mode)
            body=data.xmat[self.root].reshape(3,3);yaw=math.atan2(body[1,0],body[0,0])
            self.sync_frame(data)
            rotation=data.geom_xmat[self.head].reshape(3,3)
            expected_anchor=data.geom_xpos[self.head]+np.array([0,0,(np.abs(rotation)@self.model.geom_size[self.head])[2]+.10])
            self.max_anchor_error=max(self.max_anchor_error,float(np.linalg.norm(self.frame.position-expected_anchor)))
            self.max_heading_error=max(self.max_heading_error,abs(math.atan2(math.sin(self.frame_yaw-yaw),math.cos(self.frame_yaw-yaw))))
            if self.state.held:
                self.held_pose_samples+=1
                self.held_yaw_min=yaw if self.held_yaw_min is None else min(self.held_yaw_min,yaw)
                self.held_yaw_max=yaw if self.held_yaw_max is None else max(self.held_yaw_max,yaw)
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
                    interaction_ui_version=INTERACTION_UI_VERSION,free_drag=True,follow_camera=self.follow.value,camera_updates=self.camera_updates,
                    robot_yaw_rad=yaw,joystick_frame_yaw_rad=self.frame_yaw,joystick_anchor_world_m=list(self.frame.position),
                    joystick_tip_world_m=(np.asarray(self.frame.position)+np.array([[math.cos(yaw),-math.sin(yaw),0.],[math.sin(yaw),math.cos(yaw),0.],[0.,0.,1.]])@vector).tolist(),
                    input_frame='current_robot_body',held_deflection='body_relative',anchor_max_error_m=self.max_anchor_error,
                    heading_max_error_rad=self.max_heading_error,held_pose_samples=self.held_pose_samples,held_yaw_range_rad=[self.held_yaw_min,self.held_yaw_max],
                    camera_views=[dict(position=list(c.camera.position),look_at=list(c.camera.look_at),wxyz=list(c.camera.wxyz),fov_rad=c.camera.fov,aspect=c.camera.aspect,image_width=c.camera.image_width,image_height=c.camera.image_height) for c in self.server.get_clients().values()],
                    nonzero_command_steps=self.nonzero_command_steps,last_nonzero_command=self.last_nonzero_command.tolist(),joystick_limits=self.limits,
                    hand_distance_max=None if hands is None else float(hands.max()),
                    foot_distance_max=None if feet is None else float(feet.max()),
                    foot_distance_max_per_sensor=None if feet is None else feet.reshape(len(feet),-1).max(axis=1).tolist(),
                    tactile_kind='unchanged Week3 distance response; no force calibration',**self.performance)
                temp=self.status_path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');temp.replace(self.status_path)
