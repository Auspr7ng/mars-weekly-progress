"""Validate hold/release steering through the exact spring-centred web mapper."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse,hashlib,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import matrix_from_quat
from sunday_locomotion.env_cfg import make_env,agent_cfg
from sunday_locomotion.arrow_control import ArrowState,joystick_to_command
from sunday_locomotion.mdp import sole_geometry,foot_forces

# Each None stage is an actual mouse-release transition, not a held zero input.
STAGES=[(3,None),(8,[.15,0,0]),(11,None),(16,[.3,0,0]),(19,None),
        (24,[0,.3,0]),(27,None),(32,[0,-.3,0]),(35,None),
        (37.5,[-.15,0,0]),(40,[-.3,0,0]),(43,None),(48,[.22,.18,0]),(51,None),
        (52,[.3,0,0]),(55,None),(56,[0,.3,0]),(60,None)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--envs',type=int,default=100)
    p.add_argument('--forward',type=float,default=.15);p.add_argument('--yaw',type=float,default=.4);p.add_argument('--backward',type=float,default=.06)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    spec=json.loads((args.checkpoint.parent/'run.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()==spec['model_sha256']
    limits=dict(forward_limit=args.forward,yaw_limit=args.yaw,backward_limit=args.backward)
    torch.set_num_threads(4)
    cfg=make_env(args.envs,spec['stage'],True,revision=spec['revision'],nominal_path=args.checkpoint.parent/'dynamics/arm_home.json')
    cfg.episode_length_s=60.1;cfg.rewards={};cfg.observations['actor'].enable_corruption=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0');args.output.parent.mkdir(parents=True,exist_ok=True)
    progress=args.output.with_suffix('.progress.json');trace=[];reasons={};state=ArrowState();previous=-1;release_zero=True
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.);runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(spec['revision'])),log_dir=None,device='cuda:0')
        runner.load(str(args.checkpoint));policy=runner.get_inference_policy(device='cuda:0')
        term=env.command_manager.get_term('twist');term.override=[0,0,0];obs,_=env.reset(seed=20261117)
        d=env.scene['robot'].data;alive=np.ones(args.envs,dtype=bool)
        with torch.inference_mode():
            for step in range(3000):
                time=step*.02;k=next(i for i,(end,_) in enumerate(STAGES) if time<end);vector=STAGES[k][1]
                if k!=previous:
                    if vector is None:
                        if state.held:state.drag('end',state.vector,1)
                        else:state.release('released')
                    else:state.drag('start',[0,0,0],1);state.drag('update',vector,1)
                    previous=k
                raw,mode=joystick_to_command(state.vector,state.held,**limits)
                if vector is None:release_zero=release_zero and not raw.any() and not state.held and not state.vector.any()
                term.override=raw
                _,low=sole_geometry(env);r=matrix_from_quat(d.root_link_quat_w)
                sample=dict(time=time,stage=k,xyz=d.root_link_pos_w.cpu().numpy().copy(),velocity=d.root_link_lin_vel_b.cpu().numpy().copy(),
                            omega=d.root_link_ang_vel_b.cpu().numpy().copy(),roll=torch.atan2(r[:,2,1],r[:,2,2]).cpu().numpy().copy(),
                            low=low.cpu().numpy().copy(),force=foot_forces(env).norm(dim=-1).cpu().numpy().copy(),alive=alive.copy(),raw=raw.copy())
                trace.append(sample)
                action=policy(TensorDict(obs,batch_size=[args.envs])).clamp(-1,1);obs,_,failed,truncated,_=env.step(action)
                for name in env.termination_manager.active_terms:
                    count=int((env.termination_manager.get_term(name)&torch.as_tensor(alive,device=env.device)).sum())
                    if count:reasons[name]=reasons.get(name,0)+count
                alive&=~(failed|truncated).cpu().numpy()
                if step%250==0:progress.write_text(json.dumps(dict(state='evaluating',time_s=time,stage=k,alive=int(alive.sum())),indent=2),encoding='utf-8')
                if not alive.any():break
        a={k:np.stack([r[k] for r in trace]) for k in trace[0]};rows=[];start=0.;passed=alive.copy()
        ticks=np.zeros((args.envs,2),int);armed=np.zeros((args.envs,2),bool);landings=[]
        for low,force,live in zip(a['low'],a['force'],a['alive']):
            ticks=np.where(low>.004,ticks+1,0);armed|=ticks>=3
            landing=armed&(force>.1*cfg.sunday_mass*9.81)&(low<.003)&live[:,None];armed&=~landing;landings.append(landing)
        landings=np.array(landings)
        for k,(end,vector) in enumerate(STAGES):
            short=end-start<1.21;begin=start+(.6 if short else 1.2)
            mask=(a['time']>=begin)&(a['time']<end);ix=np.flatnonzero(mask);row=dict(stage=k,start_s=start,end_s=end,held=vector is not None,vector=vector)
            if len(ix)<2:passed[:]=False;row.update(passed=0,incomplete=True);rows.append(row);start=end;continue
            ok=a['alive'][ix].all(axis=0);v=a['velocity'][ix];om=a['omega'][ix,:,2]
            if vector is None:
                speed=np.percentile(np.linalg.norm(v[:,:,:2],axis=-1),95,axis=0);yaw=np.percentile(np.abs(om),95,axis=0)
                drift=np.linalg.norm(a['xyz'][ix[-1],:,:2]-a['xyz'][ix[0],:,:2],axis=-1)
                support=(a['force'][ix]>.1*cfg.sunday_mass*9.81).all(axis=-1).mean(axis=0)
                ok&=(speed<.025)&(yaw<.10)&(drift<.04)&(support>.85)
                row.update(raw_command_always_zero=bool(np.all(a['raw'][ix]==0)),planar_speed_p95_mean=float(speed.mean()),yaw_rate_p95_mean=float(yaw.mean()),drift_mean_m=float(drift.mean()),double_support_fraction_mean=float(support.mean()))
            else:
                command,_=joystick_to_command(vector,True,**limits);vx=v[:,:,0].mean(axis=0);wz=om.mean(axis=0)
                if abs(command[0])>.001:ok&=vx*np.sign(command[0])>=(.015 if short else (.12 if abs(command[0])>=.14 else .70*abs(command[0])))
                if abs(command[2])>.001:ok&=wz*np.sign(command[2])>=(.08 if short else .70*abs(command[2]))
                if not short:
                    count=landings[(a['time']>=start)&(a['time']<end)].sum(axis=0);ok&=count.min(axis=1)>=1
                    row['landings_mean']=count.mean(axis=0).tolist()
                row.update(command=command.tolist(),actual_vx_mean=float(vx.mean()),actual_yaw_rate_mean=float(wz.mean()))
            passed&=ok;row['passed']=int(ok.sum());rows.append(row);start=end
        result=dict(checkpoint=str(args.checkpoint),episodes=args.envs,survived=int(alive.sum()),passed=int(passed.sum()),release_zero_contract=bool(release_zero),
                    accepted=bool(release_zero and alive.mean()>=.95 and passed.mean()>=.9),termination_reasons=reasons,limits=limits,stages=rows,
                    scope='Exact body-frame hold-to-move mapper and release lifecycle; includes one-second bursts and repeat release/quiet support; real PPO feedback')
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');np.savez_compressed(args.output.with_suffix('.npz'),**a)
        progress.write_text(json.dumps(dict(state='completed',passed=result['accepted']),indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':main()
