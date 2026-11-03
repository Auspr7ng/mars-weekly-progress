"""Deterministic first-episode evaluation; automatic resets never erase failures."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from sunday_locomotion.env_cfg import make_env, agent_cfg
from sunday_locomotion.mdp import contact,sole_geometry
from mjlab.utils.lab_api.math import matrix_from_quat


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--envs',type=int,default=100)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--command',type=float,nargs=3,default=[0.,0.,0.])
    p.add_argument('--seed',type=int,default=741)
    p.add_argument('--zero-action',action='store_true',help='Evaluate the PD reference, not a learned policy')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.envs<1 or not 0<args.seconds<=20:raise ValueError('Invalid bounded evaluation size')
    checkpoint=args.checkpoint.resolve()
    spec=json.loads((checkpoint.parent/'run.json').read_text(encoding='utf-8'))
    model_hash=hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()
    if model_hash!=spec['model_sha256']:raise ValueError('Checkpoint belongs to another dynamics model')
    torch.set_num_threads(4)
    revision=spec.get('revision',1)
    nominal=checkpoint.parent/'dynamics'/('arm_home.json' if revision>=7 else ('natural_home.json' if revision>=6 else ('quality_home.json' if revision>=5 else ('gait_home.json' if revision>=2 else 'home.json'))))
    if nominal.exists() and 'nominal_sha256' in spec and hashlib.sha256(nominal.read_bytes()).hexdigest()!=spec['nominal_sha256']:
        raise ValueError('Archived nominal pose does not match checkpoint provenance')
    cfg=make_env(args.envs,spec['stage'],spec['whole_body'],gait_phase=spec['gait_phase'],revision=revision,nominal_path=nominal if nominal.exists() else None)
    cfg.seed=args.seed
    cfg.episode_length_s=args.seconds
    cfg.observations['actor'].enable_corruption=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        policy=None
        if not args.zero_action:
            runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(revision)),log_dir=None,device='cuda:0')
            runner.load(str(checkpoint))
            policy=runner.get_inference_policy(device='cuda:0')
        env.command_manager.get_term('twist').override=args.command
        obs,_=env.reset(seed=args.seed)
        data=env.scene['robot'].data
        start=data.root_link_pos_w.clone()
        end=start.clone()
        end_quat=data.root_link_quat_w.clone()
        start_rotation=matrix_from_quat(end_quat)
        start_yaw=torch.atan2(start_rotation[:,1,0],start_rotation[:,0,0])
        alive=torch.ones(args.envs,dtype=torch.bool,device=env.device)
        failed=torch.zeros_like(alive)
        completed=torch.zeros_like(alive)
        steps=torch.zeros(args.envs,device=env.device)
        squared_error=torch.zeros(args.envs,device=env.device)
        axis_squared_error=torch.zeros(args.envs,2,device=env.device)
        lateral_velocity_sum=torch.zeros_like(squared_error)
        yaw_rate_sum=torch.zeros_like(squared_error)
        switch_count=torch.zeros(args.envs,device=env.device)
        previous_contact=contact(env).clone()
        initial_foot,_=sole_geometry(env)
        last_landing=initial_foot.clone()
        air_ticks=torch.zeros(args.envs,2,device=env.device,dtype=torch.long)
        in_swing=torch.zeros(args.envs,2,device=env.device,dtype=torch.bool)
        peak_clearance=torch.zeros(args.envs,2,device=env.device)
        max_clearance=torch.zeros_like(peak_clearance)
        valid_steps=torch.zeros_like(air_ticks)
        last_step_foot=torch.full((args.envs,),-1,device=env.device,dtype=torch.long)
        alternations=torch.zeros(args.envs,device=env.device,dtype=torch.long)
        max_lateral=torch.zeros(args.envs,device=env.device)
        reasons={k:0 for k in ('tilt','height','ground','self','nonfinite','boundary')}
        with torch.inference_mode():
            for _ in range(round(args.seconds/env.step_dt)):
                before=data.root_link_pos_w.clone()
                before_quat=data.root_link_quat_w.clone()
                # Use pre-step velocity with its matching smoothed command.
                target=env.command_manager.get_command('twist')
                err=((data.root_link_lin_vel_b[:,:2]-target[:,:2])**2).sum(dim=1)
                squared_error+=err*alive
                axis_squared_error+=(data.root_link_lin_vel_b[:,:2]-target[:,:2]).square()*alive[:,None]
                lateral_velocity_sum+=data.root_link_lin_vel_b[:,1]*alive
                yaw_rate_sum+=data.root_link_ang_vel_b[:,2].abs()*alive
                steps+=alive
                action=torch.zeros(args.envs,wrapper.num_actions,device=env.device) if policy is None else policy(TensorDict(obs,batch_size=[args.envs])).clamp(-1,1)
                obs,_,term,trunc,_=env.step(action)
                done=term|trunc
                for key in reasons:
                    reasons[key]+=int((env.termination_manager.get_term(key)&alive).sum())
                # A terminal environment is already reset by mjlab. Retain its last
                # pre-reset position (at most one 20 ms interval before failure).
                end[alive]=torch.where(done[:,None],before,data.root_link_pos_w)[alive]
                end_quat[alive]=torch.where(done[:,None],before_quat,data.root_link_quat_w)[alive]
                current_contact=contact(env)
                switch_count+=((current_contact!=previous_contact).any(dim=1)&alive&~done)
                previous_contact=current_contact.clone()
                foot_pos,clearance=sole_geometry(env)
                valid=alive&~done
                airborne=clearance>.004
                air_ticks=torch.where(airborne,air_ticks+1,0)
                in_swing|=air_ticks>=3
                peak_clearance=torch.maximum(peak_clearance,clearance)
                max_clearance=torch.maximum(max_clearance,torch.where(valid[:,None],clearance,0))
                landed=in_swing&current_contact&(clearance<.002)&valid[:,None]
                forward_stride=foot_pos[...,0]-last_landing[...,0]
                step_ok=landed&(peak_clearance>.007)&(forward_stride>.015)
                valid_steps+=step_ok
                for foot in (0,1):
                    changed=step_ok[:,foot]&(last_step_foot!=foot)&(last_step_foot>=0)
                    alternations+=changed
                    last_step_foot=torch.where(step_ok[:,foot],foot,last_step_foot)
                last_landing=torch.where(landed[...,None],foot_pos,last_landing)
                peak_clearance=torch.where(landed,0,peak_clearance)
                in_swing&=~landed
                max_lateral=torch.maximum(max_lateral,torch.where(valid,(data.root_link_pos_w[:,1]-start[:,1]).abs(),0))
                failed|=alive&term
                completed|=alive&trunc&~env.termination_manager.get_term('boundary')&~term
                alive&=~done
                if not alive.any():break
        drift=torch.linalg.vector_norm((end-start)[:,:2],dim=1)
        rmse=torch.sqrt(squared_error/steps.clamp(min=1))
        end_rotation=matrix_from_quat(end_quat)
        yaw_change=torch.atan2(end_rotation[:,1,0],end_rotation[:,0,0])-start_yaw
        yaw_change=torch.atan2(torch.sin(yaw_change),torch.cos(yaw_change))
        report=dict(checkpoint=None if args.zero_action else str(checkpoint),schema_reference=str(checkpoint),
                    controller='zero-residual PD' if args.zero_action else 'deterministic PPO',model_sha256=model_hash,seed=args.seed,
                    mass_kg=cfg.sunday_mass,command=args.command,episodes=args.envs,
                    seconds=args.seconds,successes=int(completed.sum()),failures=int(failed.sum()),
                    success_rate=float(completed.float().mean()),unfinished=int(alive.sum()),
                    reasons=reasons,mean_velocity_rmse_m_s=float(rmse.mean()),
                    mean_axis_velocity_rmse_xy_m_s=torch.sqrt(axis_squared_error/steps[:,None].clamp(min=1)).mean(dim=0).tolist(),
                    mean_displacement_m=float(drift.mean()),max_displacement_m=float(drift.max()),
                    mean_forward_displacement_m=float((end-start)[:,0].mean()),
                    mean_contact_switches=float(switch_count.mean()),
                    mean_valid_steps_left_right=valid_steps.float().mean(dim=0).tolist(),
                    mean_step_alternations=float(alternations.float().mean()),
                    max_lowest_sole_clearance_m=max_clearance.max(dim=0).values.tolist(),
                    mean_max_lateral_m=float(max_lateral.mean()),max_lateral_m=float(max_lateral.max()),
                    mean_abs_yaw_change_rad=float(yaw_change.abs().mean()),
                    mean_body_lateral_velocity_m_s=float((lateral_velocity_sum/steps.clamp(min=1)).mean()),
                    mean_abs_yaw_rate_rad_s=float((yaw_rate_sum/steps.clamp(min=1)).mean()),
                    success_definition='successes/success_rate indicate survival only, not walking',
                    notes='First episode only; random initial tilt/joints retained, observation noise off. A valid step requires whole-sole clearance >4mm for 60ms, peak >7mm, landing with support, and >15mm forward foot travel. Failure displacements stop one control interval before reset. No disturbance/terrain validation.')
        if max(abs(v) for v in args.command)<1e-6:
            report['standing_gate_passed']=report['success_rate']>=.95 and report['max_displacement_m']<.03
        elif args.command[0]>.03 and abs(args.command[1])+abs(args.command[2])<1e-6:
            displacement=(end-start)[:,0]
            walking=completed&(displacement>.6*args.command[0]*args.seconds)&(valid_steps.sum(dim=1)>=4)&(valid_steps.min(dim=1).values>=1)&(alternations>=3)&(max_lateral<.15)
            report.update(walking_episodes=int(walking.sum()),walking_rate=float(walking.float().mean()),
                          walking_gate_passed=bool(walking.float().mean()>=.9 and rmse.mean()<.06),
                          gate_rates=dict(forward_distance=float((displacement>.6*args.command[0]*args.seconds).float().mean()),
                                          valid_steps=float(((valid_steps.sum(dim=1)>=4)&(valid_steps.min(dim=1).values>=1)).float().mean()),
                                          alternation=float((alternations>=3).float().mean()),
                                          lateral_lane=float((max_lateral<.15).float().mean())))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('EVALUATION '+json.dumps(report),flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
