"""Replay a real GPU policy rollout with unchanged Week3 distance heatmaps."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import webbrowser
from contextlib import nullcontext
import os

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from sunday_locomotion.env_cfg import make_env,agent_cfg
from sunday_locomotion.command_programs import PROGRAMS,command_at,duration
from week3_tactile.flexitac import FourFingerFlexiTac
from week3_tactile.feet_tactile import FootTactileEngine
from week3_tactile.tactile_display import (hand_heatmap_image,foot_heatmap_image,
                                          rotation_quat,colorize)


def sensor_panels(hand,foot,scale):
    panels=[foot_heatmap_image(foot[i],i,scale) for i in range(4)]
    feet=np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
    hands=np.concatenate([hand_heatmap_image(hand,i) for i in range(2)],axis=1)
    return feet,hands


class RolloutView:
    def __init__(self,model,data,port,speed,trained_speed=None,yaw_rate=0.,arrow_control=False,accepted=False,joystick_limits=None):
        import viser
        self.server=viser.ViserServer(host='127.0.0.1',port=port)
        self.server.scene.set_up_direction('+z')
        self.server.gui.configure_theme(show_share_button=False)
        self.arrow=None;self.save_requested=False;self.compact_sensors=arrow_control;self.saved_note=None;self.saved_until=0.
        self.server.gui.add_markdown('## Sunday A1 · RL Walking' if arrow_control else '## Sunday A1 · RL policy replay')
        self.server.gui.add_markdown((('Simulation checks passed' if accepted else 'Simulation review in progress')+' · Hand and foot distance heatmaps') if arrow_control else 'Experimental checkpoint: gait quality is under review.')
        if arrow_control:self.server.gui.add_markdown('**Hold and drag the joystick above the head.** X controls forward/back; Y steers. Release to recenter and stop.')
        if trained_speed is not None:self.server.gui.add_markdown(f'Trained command: **{trained_speed:.2f} m/s**. Other speeds are experimental.')
        self.speed=self.server.gui.add_slider('Forward command (m/s)',min=0.,max=.15,step=.01,initial_value=speed,visible=not arrow_control)
        self.yaw_rate=self.server.gui.add_slider('Turn command (rad/s)',min=-.20,max=.20,step=.01,initial_value=yaw_rate,visible=not arrow_control)
        @self.server.gui.add_button('Stop' if arrow_control else 'Stop walking (zero command)').on_click
        def stop_motion(_):
            self.speed.value=0.;self.yaw_rate.value=0.
            if self.arrow:self.arrow.stop()
        self.reset_requested=False
        self.stopped=False
        with self.server.gui.add_folder('Controls & Capture',expand_by_default=False) if arrow_control else nullcontext():
            self.paused=self.server.gui.add_checkbox('Pause simulation' if arrow_control else 'Pause',initial_value=False)
            self.status=self.server.gui.add_markdown('Loading checkpoint...',visible=not arrow_control)
            @self.server.gui.add_button('Reset episode').on_click
            def reset(_):self.reset_requested=True
            @self.server.gui.add_button('Close viewer' if arrow_control else 'Stop viewer').on_click
            def stop(_):self.stopped=True
            if arrow_control:
                follow=self.server.gui.add_checkbox('Follow camera (off: free orbit)',initial_value=True)
                @self.server.gui.add_button('Save heatmaps').on_click
                def save(_):self.save_requested=True
        if arrow_control:
            from sunday_locomotion.arrow_control import HeadArrow
            self.arrow=HeadArrow(self.server,model,data,WEEK/'results/web_demo/live_status.json',follow=follow,limits=joystick_limits)
        self.images=[]
        with self.server.gui.add_folder('Feet · 4 x 32 x 32'):
            self.scale=self.server.gui.add_markdown('Shared foot color scale')
            if arrow_control:self.images.append(self.server.gui.add_image(sensor_panels(np.zeros((4,12,32)),np.zeros((4,32,32)),.005)[0],label=''))
            else:
                for i in range(4):self.images.append(self.server.gui.add_image(foot_heatmap_image(np.zeros((32,32)),i),label=['Left toe','Left heel','Right toe','Right heel'][i]))
        with self.server.gui.add_folder('Hands · 4 x 12 x 32'):
            if arrow_control:self.images.append(self.server.gui.add_image(sensor_panels(np.zeros((4,12,32)),np.zeros((4,32,32)),.005)[1],label=''))
            else:
                for i in range(2):self.images.append(self.server.gui.add_image(hand_heatmap_image(np.zeros((4,12,32)),i),label=['Left hand','Right hand'][i]))
        self.meshes=[]
        for gid in range(model.ngeom):
            if model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[gid]!=1:continue
            mid=model.geom_dataid[gid]
            va,vn=model.mesh_vertadr[mid],model.mesh_vertnum[mid]
            fa,fn=model.mesh_faceadr[mid],model.mesh_facenum[mid]
            h=self.server.scene.add_mesh_simple(f'/robot/{gid}',vertices=np.array(model.mesh_vert[va:va+vn]),
                faces=np.array(model.mesh_face[fa:fa+fn]),color=(185,194,204),side='double')
            self.meshes.append((gid,h))
        self.server.scene.add_grid('/ground',width=20,height=20,plane='xy',cell_size=.1,section_size=1.)
        self.points=self.server.scene.add_point_cloud('/taxels',points=np.zeros((5632,3)),colors=np.zeros((5632,3),dtype=np.uint8),point_size=.0015)
        @self.server.on_client_connect
        def camera(client):
            center=data.qpos[:3]+np.array([0,0,.05])
            client.camera.up_direction=(0,0,1)
            client.camera.position=center+np.array([.60,-.80,.60] if arrow_control else [.8,-1.05,.48])
            client.camera.look_at=center
            if self.arrow:self.arrow.cache_camera(client,data)

    def update(self,data,hands,feet,label):
        hand=hands.response[0].cpu().numpy()
        foot=feet.response[0].cpu().numpy()
        scale=max(.005,float(foot.max()))
        with self.server.atomic():
            for gid,h in self.meshes:
                h.position=data.geom_xpos[gid]
                h.wxyz=rotation_quat(data.geom_xmat[gid])
            points=torch.cat([hands.tactile_points_w_per_sensor[0],feet.tactile_points_w_per_sensor[0]],dim=1).cpu().numpy().reshape(-1,4)
            self.points.points=points[:,:3]
            self.points.colors=colorize(points[:,3])
            if self.compact_sensors:
                foot_panel,hand_panel=sensor_panels(hand,foot,scale)
                self.images[0].image=foot_panel;self.images[1].image=hand_panel
            else:
                for i in range(4):self.images[i].image=foot_heatmap_image(foot[i],i,scale)
                for i in range(2):self.images[4+i].image=hand_heatmap_image(hand,i)
            self.scale.content=f'Shared scale: 0 – {scale:.4f} (auto)' if self.arrow else f'Foot color range: 0 to {scale:.4f} (automatic, same scale on all four maps)'
            if self.arrow:
                self.status.visible=bool(self.saved_note and time.monotonic()<self.saved_until)
                if self.status.visible:self.status.content=self.saved_note
            else:self.status.content=label
            if self.arrow:self.arrow.update(data,hand,foot)
        if self.save_requested:
            self.save_requested=False
            from datetime import datetime
            from PIL import Image
            destination=WEEK/'results/web_demo'/datetime.now().strftime('capture_%Y%m%d_%H%M%S')
            destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/'distance_arrays.npz',hands=hand,feet=foot,root_qpos=data.qpos.copy())
            for i in range(4):Image.fromarray(foot_heatmap_image(foot[i],i,scale)).save(destination/f'foot_{i}.png')
            for i in range(2):Image.fromarray(hand_heatmap_image(hand,i)).save(destination/f'hand_{i}_two_pads.png')
            self.saved_note=f'Saved heatmaps: {destination.name}';self.saved_until=time.monotonic()+4.
            self.status.content=self.saved_note;self.status.visible=True


def video_frame(renderer,camera,native,hands,feet,label,height_offset=.04,dual_view=False):
    from PIL import Image,ImageDraw
    camera.lookat[:]=native.qpos[:3]+np.array([0,0,height_offset])
    option=mujoco.MjvOption()
    option.geomgroup[:]=0
    option.geomgroup[0]=option.geomgroup[1]=1
    canvas=Image.new('RGB',(1280,720),'#f7f9fc')
    if dual_view:
        for azimuth,left in ((0,0),(90,430)):
            camera.azimuth=azimuth
            renderer.update_scene(native,camera=camera,scene_option=option)
            canvas.paste(Image.fromarray(renderer.render()),(left,0))
    else:
        renderer.update_scene(native,camera=camera,scene_option=option)
        canvas.paste(Image.fromarray(renderer.render()),(0,0))
    draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,1280,46),fill='#172032')
    draw.text((12,14),label,fill='white')
    if dual_view:
        draw.text((12,54),'FRONT / same rollout',fill='white')
        draw.text((442,54),'SIDE / same rollout',fill='white')
    foot=feet.response[0].cpu().numpy()
    hand=hands.response[0].cpu().numpy()
    scale=max(.005,float(foot.max()))
    draw.text((875,53),f'Distance heatmaps | foot scale 0..{scale:.4f}',fill='#172032')
    for i in range(4):
        canvas.paste(Image.fromarray(foot_heatmap_image(foot[i],i,scale)).resize((200,215)),(870+(i%2)*205,78+(i//2)*220))
    for i in range(2):
        canvas.paste(Image.fromarray(hand_heatmap_image(hand,i)).resize((200,204)),(870+i*205,514))
    return np.asarray(canvas)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--speed',type=float)
    p.add_argument('--yaw-rate',type=float,default=0.)
    p.add_argument('--latest-training',action='store_true',help='Replay the active experimental run, preserving the selected baseline')
    p.add_argument('--accepted-policy',action='store_true',help='Load the reviewed command policy from its manifest')
    p.add_argument('--arrow-control',action='store_true',help='Drag a head-mounted world direction/speed arrow')
    p.add_argument('--program',choices=list(PROGRAMS),help='Record a repeatable speed/stop/turn command program')
    p.add_argument('--port',type=int,default=8084)
    p.add_argument('--open-browser',action='store_true')
    p.add_argument('--viewer-duration',type=float,default=0.,help='Optional wall-clock limit for the interactive viewer; 0 keeps it open')
    p.add_argument('--azimuth',type=float,default=135.)
    p.add_argument('--legs-closeup',action='store_true')
    p.add_argument('--dual-view',action='store_true',help='Render synchronized front/side views of the same physical rollout')
    p.add_argument('--record',type=Path,help='Record a bounded rollout instead of opening the viewer')
    p.add_argument('--seconds',type=float)
    args=p.parse_args()
    accepted_path=WEEK/'config/accepted_policy.json'
    if args.accepted_policy:
        accepted_manifest=json.loads(accepted_path.read_text(encoding='utf-8'))
        if not accepted_manifest.get('accepted'):raise ValueError('Command policy is not accepted yet')
        args.checkpoint=WEEK/accepted_manifest['checkpoint']
    if args.arrow_control:args.speed=0.
    if args.program and not args.record:raise ValueError('Command programs require --record; the viewer uses its live sliders')
    if args.seconds is None:args.seconds=duration(args.program) if args.program else 10.
    if (args.speed is not None and not 0<=args.speed<=.15) or not 0<args.seconds<=60 or abs(args.yaw_rate)>.3:raise ValueError('Invalid replay command or duration')
    if args.checkpoint is None:
        active_path=WEEK/'config/active_run.json'
        active=json.loads(active_path.read_text()) if active_path.exists() else {}
        if active.get('selected_checkpoint') and not args.latest_training:
            args.checkpoint=(WEEK/active['selected_checkpoint']).resolve()
        else:
            status_path=WEEK/active['status_path'] if active else WEEK/'logs/forward_measured_r1/status.json'
            status=json.loads(status_path.read_text())
            args.checkpoint=Path(status['checkpoint'])
    spec=json.loads((args.checkpoint.parent/'run.json').read_text())
    accepted_manifest=json.loads(accepted_path.read_text(encoding='utf-8')) if accepted_path.exists() else {}
    accepted=bool(accepted_manifest.get('accepted') and accepted_manifest.get('checkpoint_sha256')==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    model_path=args.checkpoint.parent/'dynamics/sunday_a1_locomotion.xml'
    if not model_path.exists():model_path=WEEK/'model/sunday_a1_locomotion.xml'
    actual_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if spec['model_sha256']!=actual_hash:raise ValueError('Checkpoint model does not match its archived dynamics')
    torch.set_num_threads(4)
    revision=spec.get('revision',1)
    trained_speed=.08 if revision>=2 and spec['stage']=='forward' else None
    if revision>=8 and spec['stage']=='forward':trained_speed=.06
    elif revision>=7 and spec['stage']=='forward':
        import re
        match=re.match(r'model_(\d+)',args.checkpoint.stem)
        iteration=int(match.group(1)) if match else 0
        trained_speed=.02+.08*max(0.,min(1.,(iteration-100)/200))
    if args.speed is None:args.speed=trained_speed if trained_speed is not None else (.06 if revision>=12 else .1)
    nominal=args.checkpoint.parent/'dynamics'/('arm_home.json' if revision>=7 else ('natural_home.json' if revision>=6 else ('quality_home.json' if revision>=5 else ('gait_home.json' if revision>=2 else 'home.json'))))
    if nominal.exists() and 'nominal_sha256' in spec and hashlib.sha256(nominal.read_bytes()).hexdigest()!=spec['nominal_sha256']:
        raise ValueError('Archived nominal pose does not match checkpoint provenance')
    cfg=make_env(1,spec['stage'],spec['whole_body'],play=True,gait_phase=spec['gait_phase'],revision=revision,nominal_path=nominal if nominal.exists() else None,model_path=model_path)
    cfg.rewards={}
    # Playback does not use training rewards. Command/state/physics updates remain active.
    cfg.rewards={}
    if args.program:cfg.episode_length_s=max(cfg.episode_length_s,args.seconds+.04)
    elif not args.record:cfg.episode_length_s=3600.
    cfg.auto_reset=False
    if args.arrow_control:
        cfg.terminations.pop('boundary',None)
        cfg.terminations.pop('time_out',None)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    renderer=writer=ui=None
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(revision)),log_dir=None,device='cuda:0')
        runner.load(str(args.checkpoint))
        policy=runner.get_inference_policy(device='cuda:0')
        twist=env.command_manager.get_term('twist')
        twist.override=command_at(args.program,0) if args.program else [args.speed,0.,args.yaw_rate]
        obs,_=env.reset(seed=741)
        model=env.sim.mj_model
        native=mujoco.MjData(model)
        def copy_pose():
            native.qpos[:]=env.sim.data.qpos[0].cpu().numpy()
            native.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(model,native)
        copy_pose()
        hands=FourFingerFlexiTac(model,prefix='robot/')
        feet=FootTactileEngine(model,prefix='robot/')
        if args.record:
            import imageio_ffmpeg
            args.record.parent.mkdir(parents=True,exist_ok=True)
            render_width=430 if args.dual_view else 860
            model.vis.global_.offwidth,model.vis.global_.offheight=render_width,720
            renderer=mujoco.Renderer(model,height=720,width=render_width)
            camera=mujoco.MjvCamera()
            camera.distance,camera.azimuth,camera.elevation=.60 if args.legs_closeup else 1.1,args.azimuth,-15
            writer=imageio_ffmpeg.write_frames(str(args.record),size=(1280,720),fps=25,codec='libx264',macro_block_size=1)
            writer.send(None)
        else:
            ui=RolloutView(model,native,args.port,args.speed,trained_speed=trained_speed,yaw_rate=args.yaw_rate,arrow_control=args.arrow_control,accepted=accepted,joystick_limits=accepted_manifest.get('joystick_limits'))
            print(f'WEB_DEMO_READY http://127.0.0.1:{args.port}',flush=True)
            if args.arrow_control:
                info=dict(port=args.port,pid=os.getpid(),checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),accepted_policy=accepted,url=f'http://127.0.0.1:{args.port}',interaction_mode='hold_to_move_joystick',joystick_limits=ui.arrow.limits,mapper_version='stable_floor_v1',physics_backend='native_mujoco',actor_device='cpu')
                (WEEK/'results/web_demo/server_info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
            if args.open_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
        if ui and args.arrow_control:
            from sunday_locomotion.native_web_player import run_native_web
            cpu_policy=runner.get_inference_policy(device='cpu')
            run_native_web(model,native,env,cpu_policy,ui,hands,feet,f'{args.checkpoint.parent.name}/{args.checkpoint.name}',args.viewer_duration)
            return
        step=frames=0
        performance_time=time.perf_counter();performance_step=0
        done=False
        trace=[]
        view_started=time.perf_counter()
        with torch.inference_mode():
            while True:
                start=time.perf_counter()
                if ui:
                    if args.viewer_duration>0 and start-view_started>=args.viewer_duration:break
                    if ui.stopped:break
                    if ui.reset_requested:
                        if ui.arrow:ui.arrow.stop()
                        obs,_=env.reset();ui.reset_requested=False;ui.paused.value=False;step=0;done=False
                        performance_time=time.perf_counter();performance_step=step
                    if ui.paused.value or done:
                        performance_time=time.perf_counter();performance_step=step
                        time.sleep(.02);continue
                    twist.override=ui.arrow.command(native) if ui.arrow else [float(ui.speed.value),0.,float(ui.yaw_rate.value)]
                elif args.program:twist.override=command_at(args.program,step*.02)
                action=policy(TensorDict(obs,batch_size=[1])).clamp(-1,1)
                obs,_,term,trunc,_=env.step(action)
                step+=1
                if ui and ui.arrow:
                    now=time.perf_counter();elapsed=now-performance_time
                    ui.arrow.performance['simulation_time_s']=step*.02
                    if elapsed>=1.:
                        ui.arrow.performance['real_time_factor']=(step-performance_step)*.02/elapsed
                        performance_time=now;performance_step=step
                done=bool((term|trunc).item())
                reasons=[name for name in env.termination_manager.active_terms if env.termination_manager.get_term(name).any()] if done else []
                copy_pose()
                velocity=float(env.scene['robot'].data.root_link_lin_vel_b[0,0])
                target=float(twist.command[0,0])
                yaw_target=float(twist.command[0,2]);yaw_velocity=float(env.scene['robot'].data.root_link_ang_vel_b[0,2])
                label=f'{args.checkpoint.parent.name}/{args.checkpoint.name} | t={step*.02:.2f}s | v={velocity:.2f}/{target:.2f}m/s | yaw={yaw_velocity:.2f}/{yaw_target:.2f}rad/s'
                if done:label+=' | END: '+','.join(reasons)
                if args.record:trace.append(dict(t=step*.02,x=float(native.qpos[0]),y=float(native.qpos[1]),z=float(native.qpos[2]),vx=velocity,target=target,wz=yaw_velocity,yaw_target=yaw_target,action=action[0].cpu().tolist()))
                if step%2==0 or done:
                    hand=hands.sample_mujoco(native)
                    foot=feet.sample_mujoco(native)
                    if ui:ui.update(native,hand,foot,label)
                    else:
                        if args.legs_closeup:label+=' | LEG CLOSE-UP'
                        frame=video_frame(renderer,camera,native,hand,foot,label,height_offset=-.10 if args.legs_closeup else .04,dual_view=args.dual_view)
                        writer.send(frame);frames+=1
                        if frames==1:
                            from PIL import Image
                            Image.fromarray(frame).save(args.record.with_suffix('.png'))
                if ui:
                    if done:ui.paused.value=True
                    time.sleep(max(0.,.02-(time.perf_counter()-start)))
                elif done or step>=round(args.seconds/.02):break
        if args.record:
            report=dict(checkpoint=str(args.checkpoint),model_sha256=actual_hash,command_m_s=None if args.program else args.speed,command_program=args.program,yaw_rate_command=args.yaw_rate,
                        duration_s=step*.02,frames=frames,dual_view=args.dual_view,termination_reasons=reasons,trace=trace,
                        notes='Actual deterministic policy rollout in GPU physics. CPU data is only a render/sensor pose copy. Distance heatmaps are not calibrated pressure.')
            args.record.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(f'RECORDED {args.record} ({frames} frames)',flush=True)
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
        if ui is not None:ui.server.stop()
        env.close()


if __name__=='__main__':main()
