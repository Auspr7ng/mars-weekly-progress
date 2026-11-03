"""Realtime native MuJoCo/CPU-policy playback with the same action contract."""
import time
import math
import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from .native_observations import NativeHistory
from .action_ranges import native_scales
from .control_mdp import slew_rates


def run_native_web(model,data,env,policy,ui,hands,feet,checkpoint,viewer_duration=0):
    revision=env.cfg.sunday_revision;robot=env.scene['robot'];action=env.action_manager.get_term('residual')
    names=list(robot.joint_names);ids=action.ids.cpu().numpy();base_scales=action.scales.cpu().numpy();speeds=action.speeds.cpu().numpy()
    default=robot.data.default_joint_pos[0].cpu().numpy().copy();limits=robot.data.soft_joint_pos_limits[0].cpu().numpy()
    actuator_joints=[names.index(model.actuator(i).name.removeprefix('robot/')) for i in range(model.nu)]
    foot_bodies=[model.body('robot/'+s+'_foot_link_1').id for s in ('l','r')]
    root=model.body('robot/base_link').id;q0=data.qpos.copy();v0=data.qvel.copy()
    rate=np.asarray(slew_rates(revision))*.02;started=time.perf_counter()
    def reset():
        data.qpos[:]=q0;data.qvel[:]=v0;data.ctrl[:]=default[actuator_joints];data.time=0.
        mujoco.mj_forward(model,data)
        return NativeHistory(model,default,names,len(ids),revision),np.zeros(3),0.,np.zeros(len(ids)),default.copy(),dict(tilt=0,height=0,ground=0,self=0)
    history,command,heading,last,targets,counters=reset();step=0;done=False;perf_time=time.perf_counter();perf_step=0
    with torch.inference_mode():
        while not ui.stopped:
            start=time.perf_counter()
            if viewer_duration>0 and start-started>=viewer_duration:break
            if ui.reset_requested:
                ui.arrow.stop();history,command,heading,last,targets,counters=reset()
                step=0;done=False;ui.reset_requested=False;ui.paused.value=False;perf_time=start;perf_step=0
            if ui.paused.value or done:
                perf_time=start;perf_step=step;time.sleep(.02);continue
            requested=ui.arrow.command(data)
            observation=history.push(data,command,last,step,heading)
            raw=policy(TensorDict({'actor':torch.as_tensor(observation[None],device='cpu')},batch_size=[1]))[0]
            if not torch.isfinite(raw).all():raise FloatingPointError('Nonfinite native policy output')
            out=raw.clamp(-1,1).numpy()
            target=default[ids]+out*native_scales(base_scales,action.names,command,revision)
            target=targets[ids]+.3*(target-targets[ids])
            target=np.clip(target,targets[ids]-speeds*.02,targets[ids]+speeds*.02)
            targets[ids]=np.clip(target,limits[ids,0],limits[ids,1]);data.ctrl[:]=targets[actuator_joints]
            bad_ground=False;bad_self=False
            for _ in range(10):
                mujoco.mj_step(model,data)
                for ci in range(data.ncon):
                    c=data.contact[ci];b1=int(model.geom_bodyid[c.geom1]);b2=int(model.geom_bodyid[c.geom2])
                    plane=model.geom_type[c.geom1]==mujoco.mjtGeom.mjGEOM_PLANE or model.geom_type[c.geom2]==mujoco.mjtGeom.mjGEOM_PLANE
                    if plane and b1 not in foot_bodies and b2 not in foot_bodies:
                        force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
                        bad_ground|=abs(force[0])>.02*env.cfg.sunday_mass*9.81
                    if not plane and c.dist<0 and model.geom(c.geom1).name.startswith('robot/') and model.geom(c.geom2).name.startswith('robot/'):bad_self=True
            mujoco.mj_forward(model,data);step+=1;last=out.copy()
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise FloatingPointError('Nonfinite native state')
            command+=np.clip(requested-command,-rate,rate);heading+=command[2]*.02
            cos=float(data.xmat[root].reshape(3,3)[2,2])
            bad=dict(tilt=cos<.5,height=data.qpos[2]<.55*env.cfg.sunday_h0,ground=bad_ground,self=bad_self)
            for name,flag in bad.items():counters[name]=counters[name]+1 if flag else 0
            reasons=[name for name,count in counters.items() if count>=(5 if name in ('height','tilt') else 2)]
            if cos<.173648:reasons.append('extreme_tilt')
            done=bool(reasons)
            if done:ui.arrow.stop();ui.arrow.command(data)
            now=time.perf_counter();ui.arrow.performance.update(simulation_time_s=step*.02,physics_backend='native_mujoco',actor_device='cpu')
            if now-perf_time>=1.:
                ui.arrow.performance['real_time_factor']=(step-perf_step)*.02/(now-perf_time);perf_time=now;perf_step=step
            if step%2==0 or done:
                velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,root,velocity,1)
                label=f'{checkpoint} | t={step*.02:.2f}s | v={velocity[3]:.2f}/{command[0]:.2f}m/s | yaw={velocity[2]:.2f}/{command[2]:.2f}rad/s'
                if done:label+=' | END: '+','.join(reasons)
                ui.update(data,hands.sample_mujoco(data),feet.sample_mujoco(data),label)
            if done:ui.paused.value=True
            time.sleep(max(0.,.02-(time.perf_counter()-start)))
