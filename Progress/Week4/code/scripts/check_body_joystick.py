"""Regression for held joystick tracking, live heading and pointer isolation.

Uses real MuJoCo poses and the actual drag callback; scene handles are recorded
locally so the check cannot operate the user's live viewer.
"""
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np

WEEK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WEEK/'code/src'))
from sunday_locomotion.arrow_control import HeadArrow


class Node:
    def __init__(self,**props):
        self.position=np.zeros(3);self.wxyz=(1.,0.,0.,0.)
        self.__dict__.update(props)
    def on_drag(self,callback):self.callback=callback;return callback


class Scene:
    def __getattr__(self,name):
        if not name.startswith('add_'):raise AttributeError(name)
        return lambda *args,**props:Node(**props)


def camera_at(position,look_at):
    position=np.asarray(position,dtype=float);look_at=np.asarray(look_at,dtype=float)
    view=look_at-position;view/=np.linalg.norm(view)
    right=np.cross(view,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(view,right);rotation=np.stack([right,down,view],axis=1)
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,rotation.ravel())
    return SimpleNamespace(position=position,look_at=look_at,wxyz=quat,fov=math.pi/3,aspect=1.6,image_width=1600,image_height=1000)


async def main():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="robot/base_link" pos="0 0 .24"><freejoint/>
        <geom type="box" size=".08 .06 .08" mass="1"/>
        <body pos="0 0 .25"><geom name="robot/collision_head_1" type="box" size=".06 .025 .06" mass=".1"/></body>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    client=SimpleNamespace(camera=camera_at([2.,-2.,3.],[0.,0.,.5]))
    clients={1:client}
    server=SimpleNamespace(scene=Scene(),gui=SimpleNamespace(add_markdown=lambda *a:Node()),get_clients=lambda:clients)
    arrow=HeadArrow(server,model,data,WEEK/'results/validation/body_follow_v3_test_status.json',follow=SimpleNamespace(value=False),limits=dict(forward_limit=.12,yaw_limit=.25,backward_limit=.06))
    pointer=np.array([.5,.5]);poses=[]

    def pose(yaw,xy):
        data.qpos[:3]=[xy[0],xy[1],.24]
        data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        mujoco.mj_forward(model,data);arrow.update(data,None,None)

    async def event(phase,new_pointer):
        nonlocal pointer
        await arrow.tip.callback(SimpleNamespace(phase=phase,client=client,client_id=1,
            start_screen_pos=pointer.copy(),end_screen_pos=np.array(new_pointer)))
        pointer=np.array(new_pointer)

    async def move_body(delta):
        # Project a known ground-plane movement onto the current camera.
        yaw=math.atan2(data.xmat[arrow.root].reshape(3,3)[1,0],data.xmat[arrow.root].reshape(3,3)[0,0])
        c,s=math.cos(yaw),math.sin(yaw)
        world=np.array([c*delta[0]-s*delta[1],s*delta[0]+c*delta[1],0.])
        r=np.empty(9);mujoco.mju_quat2Mat(r,client.camera.wxyz);r=r.reshape(3,3)
        body_rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        current_point=np.asarray(arrow.frame.position)+body_rotation@arrow.state.vector
        current_camera=r.T@(current_point-client.camera.position)
        next_camera=r.T@(current_point+world-client.camera.position)
        image_size=2.*math.tan(client.camera.fov/2)*np.array([client.camera.aspect,1.])
        screen=(next_camera[:2]/next_camera[2]-current_camera[:2]/current_camera[2])/image_size
        await event('update',pointer+screen)

    await event('start',pointer)
    await move_body([.10,0.])
    initial_command=np.array(arrow.command(data));assert initial_command[0]>.03 and abs(initial_command[2])<1e-9
    for yaw,xy in [(0.,[0.,0.]),(math.pi/2,[.4,-.2]),(math.pi,[.7,.3]),(-math.pi/2,[-.3,.5])]:
        before=arrow.state.vector.copy();pose(yaw,xy)
        expected=data.geom_xpos[arrow.head]+[0.,0.,.16]
        np.testing.assert_allclose(arrow.frame.position,expected,atol=1e-12)
        r=np.empty(9);mujoco.mju_quat2Mat(r,np.asarray(arrow.frame.wxyz))
        np.testing.assert_allclose(r.reshape(3,3)[:,0],[math.cos(yaw),math.sin(yaw),0.],atol=1e-12)
        await event('update',pointer)
        np.testing.assert_allclose(arrow.state.vector,before,atol=1e-12)
        np.testing.assert_allclose(arrow.command(data),initial_command,atol=1e-12)
        await move_body([.02,0.])
        np.testing.assert_allclose(arrow.state.vector,before+[.02,0.,0.],atol=1e-9)
        arrow.state.set_vector(before)
        poses.append(dict(yaw_rad=yaw,root_xy_m=xy,anchor=list(arrow.frame.position),forward_world=r.reshape(3,3)[:,0].tolist()))
    # Camera orbit and robot motion alone must not act like joystick input.
    before=arrow.state.vector.copy();client.camera=camera_at([-2.,-2.,3.],data.qpos[:3]+[0.,0.,.3])
    await event('update',pointer);np.testing.assert_allclose(arrow.state.vector,before,atol=1e-12)
    # An independently updated orbit target must not skew the rendered basis.
    client.camera.look_at=np.array([10.,20.,30.])
    await move_body([.02,0.]);np.testing.assert_allclose(arrow.state.vector,before+[.02,0.,0.],atol=1e-9)
    arrow.state.set_vector(before)
    await move_body([0.,.06]);assert arrow.command(data)[2]>.08
    await event('end',pointer);assert not arrow.state.held and not arrow.state.vector.any() and not any(arrow.command(data))
    await event('start',pointer);await move_body([.08,0.]);clients.clear()
    assert not any(arrow.command(data)) and not arrow.state.held
    report=dict(passed=True,real_mujoco_pose_checks=poses,live_yaw_pointer_conversion=True,
        held_anchor_follows=True,held_axes_follow=True,stationary_pointer_isolated_from_robot_and_camera=True,
        body_command_retained_through_turn=True,projection_uses_rendered_camera_quaternion=True,
        off_center_perspective_roundtrip=True,release_zero=True,disconnect_zero=True)
    (WEEK/'results/validation/body_follow_v3_contract.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':asyncio.run(main())
