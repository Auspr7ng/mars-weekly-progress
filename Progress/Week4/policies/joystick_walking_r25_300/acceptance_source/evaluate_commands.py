"""Batch tests of real policy command transitions; resets never erase a failure."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse,hashlib,json
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import matrix_from_quat
from sunday_locomotion.env_cfg import make_env,agent_cfg
from sunday_locomotion.mdp import foot_forces,sole_geometry
from sunday_locomotion.command_programs import PROGRAMS,command_at,duration
from sunday_locomotion.control_metrics import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--envs',type=int,default=32);p.add_argument('--programs',nargs='+',choices=list(PROGRAMS),default=['idle','steady','speed_stop','turns','in_place'])
    p.add_argument('--seed',type=int,default=20261106);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compute-rewards',action='store_true',help='Optional diagnostic; policy evaluation does not use training rewards')
    args=p.parse_args()
    if not 1<=args.envs<=512:raise ValueError('Bounded evaluation requires 1..512 environments')
    run=args.checkpoint.parent;spec=json.loads((run/'run.json').read_text());revision=spec['revision']
    if revision<11:raise ValueError('Control checks require the full R11/R12 observation schema')
    if hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()!=spec['model_sha256']:raise ValueError('Physical model changed')
    nominal=run/'dynamics/arm_home.json'
    if hashlib.sha256(nominal.read_bytes()).hexdigest()!=spec['nominal_sha256']:raise ValueError('Nominal pose changed')
    torch.set_num_threads(4)
    cfg=make_env(args.envs,spec['stage'],spec['whole_body'],revision=revision,nominal_path=nominal)
    cfg.episode_length_s=max(duration(x) for x in args.programs)+.04;cfg.observations['actor'].enable_corruption=False
    if not args.compute_rewards:cfg.rewards={}
    env=ManagerBasedRlEnv(cfg,device='cuda:0');reports=[]
    args.output.parent.mkdir(parents=True,exist_ok=True);progress=args.output.with_suffix('.progress.json')
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.);runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(revision)),log_dir=None,device='cuda:0')
        runner.load(str(args.checkpoint));policy=runner.get_inference_policy(device='cuda:0');d=env.scene['robot'].data
        foot_ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
        for k,program in enumerate(args.programs):
            progress.write_text(json.dumps(dict(state='evaluating',program=program,checkpoint=str(args.checkpoint)),indent=2),encoding='utf-8')
            term=env.command_manager.get_term('twist');term.override=command_at(program,0)
            # Sensor/history tensors are updated inside inference_mode by env.step.
            # Reset them in the same mode on every subsequent test program.
            with torch.inference_mode():obs,_=env.reset(seed=args.seed+k)
            alive=torch.ones(args.envs,dtype=torch.bool,device=env.device);trace=[];yaw_sum=torch.zeros(args.envs,device=env.device)
            rot=matrix_from_quat(d.root_link_quat_w);previous=torch.atan2(rot[:,1,0],rot[:,0,0]);reasons={}
            with torch.inference_mode():
                for step in range(round(duration(program)/env.step_dt)):
                    time=step*env.step_dt;term.override=command_at(program,time)
                    rot=matrix_from_quat(d.root_link_quat_w);yaw=torch.atan2(rot[:,1,0],rot[:,0,0]);delta=yaw-previous
                    yaw_sum+=torch.atan2(torch.sin(delta),torch.cos(delta))*alive;previous=yaw.clone()
                    _,low=sole_geometry(env);err=term.target_yaw-yaw
                    sample=dict(time=torch.full((args.envs,),time,device=env.device),xyz=d.root_link_pos_w.clone(),
                                      velocity=d.root_link_lin_vel_b.clone(),omega=d.root_link_ang_vel_b.clone(),command=term.value.clone(),
                                      heading_error=torch.atan2(torch.sin(err),torch.cos(err)),yaw_unwrapped=yaw_sum.clone(),
                                      sole_low=low.clone(),foot_force=foot_forces(env).norm(dim=-1),
                                      foot_speed=d.site_lin_vel_w[:,foot_ids,:2].norm(dim=-1),alive=alive.clone())
                    sample['sole_position']=d.site_pos_w[:,foot_ids].clone()
                    sample['sole_velocity']=d.site_lin_vel_w[:,foot_ids].clone()
                    if trace:
                        for name in sample:
                            if name in ('time','command','alive'):continue
                            mask=alive.reshape((args.envs,)+(1,)*(sample[name].ndim-1))
                            sample[name]=torch.where(mask,sample[name],trace[-1][name])
                    trace.append(sample)
                    mean=policy(TensorDict(obs,batch_size=[args.envs]));action=mean.clamp(-1,1)
                    # Retain what the network requested and what physics received.
                    # These diagnostics do not alter either policy or actuation.
                    sample['policy_mean']=mean.clone();sample['action']=action.clone()
                    obs,_,failed,truncated,_=env.step(action)
                    for name in env.termination_manager.active_terms:
                        count=int((env.termination_manager.get_term(name)&alive).sum())
                        if count:reasons[name]=reasons.get(name,0)+count
                    alive&=~(failed|truncated)
                    if not alive.any():break
            a={key:torch.stack([s[key] for s in trace]).cpu().numpy() for key in trace[0]};a['time']=a['time'][:,0]
            # Include a failure in the final interval in survival accounting.
            a['alive'][-1]&=alive.cpu().numpy()
            report=summarize(a,program,cfg.sunday_mass);report.update(termination_reasons=reasons,planned_seconds=duration(program));reports.append(report)
            np.savez_compressed(args.output.with_name(args.output.stem+'_'+program+'_trace.npz'),**a)
            print('PROGRAM_RESULT '+json.dumps(report),flush=True)
        result=dict(checkpoint=str(args.checkpoint.resolve()),revision=revision,seed=args.seed,simulation_only=True,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),model_sha256=spec['model_sha256'],nominal_sha256=spec['nominal_sha256'],
                    action_order=env.action_manager.get_term('residual').names,
                    training_rewards_computed=args.compute_rewards,
                    command_programs=args.programs,programs=reports,all_requested_programs_passed=all(r['engineering_passed'] for r in reports),
                    complete_control_suite=set(args.programs)>={'steady','speed_stop','turns','in_place'},visual_review_required=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        progress.write_text(json.dumps(dict(state='completed',output=str(args.output)),indent=2),encoding='utf-8')
    except BaseException as exc:
        progress.write_text(json.dumps(dict(state='failed',checkpoint=str(args.checkpoint),error=repr(exc)),indent=2),encoding='utf-8')
        raise
    finally:env.close()


if __name__=='__main__':main()
