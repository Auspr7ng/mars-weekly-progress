"""Exercise the exact web arrow mapper in real policy/GPU feedback."""
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
from sunday_locomotion.arrow_control import vector_to_command
from sunday_locomotion.mdp import sole_geometry,foot_forces

STAGES=[(3.,[0.,0.,0.]),(9.,[.12,0.,0.]),(15.,[.30,0.,0.]),(20.,[0.,0.,0.]),
        (47.,[0.,.22,0.]),(51.,[0.,0.,0.]),(96.,[0.,-.22,0.]),(102.,[0.,0.,0.])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--envs',type=int,default=100);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    spec=json.loads((args.checkpoint.parent/'run.json').read_text(encoding='utf-8'));nominal=args.checkpoint.parent/'dynamics/arm_home.json'
    assert hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()==spec['model_sha256']
    torch.set_num_threads(4)
    cfg=make_env(args.envs,spec['stage'],True,revision=spec['revision'],nominal_path=nominal)
    cfg.episode_length_s=102.1;cfg.rewards={};cfg.observations['actor'].enable_corruption=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0');traces=[];reasons={};args.output.parent.mkdir(parents=True,exist_ok=True)
    progress=args.output.with_suffix('.progress.json')
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.);runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(spec['revision'])),log_dir=None,device='cuda:0');runner.load(str(args.checkpoint));policy=runner.get_inference_policy(device='cuda:0')
        term=env.command_manager.get_term('twist');term.override=[0,0,0];obs,_=env.reset(seed=20261113)
        alive=np.ones(args.envs,dtype=bool);d=env.scene['robot'].data
        with torch.inference_mode():
            for i in range(5100):
                time=i*.02;stage=next(k for k,(end,_) in enumerate(STAGES) if time<end)
                vector=STAGES[stage][1];r=matrix_from_quat(d.root_link_quat_w)
                yaw=torch.atan2(r[:,1,0],r[:,0,0]).cpu().numpy()
                mapped=[vector_to_command(vector,float(y)) for y in yaw]
                command=np.stack([x[0] for x in mapped]);term.override=command
                _,low=sole_geometry(env)
                traces.append(dict(time=time,stage=stage,xyz=d.root_link_pos_w.cpu().numpy().copy(),
                    velocity=d.root_link_lin_vel_b.cpu().numpy().copy(),omega=d.root_link_ang_vel_b.cpu().numpy().copy(),
                    heading_error=np.array([x[1]['heading_error'] for x in mapped]),command=command,
                    low=low.cpu().numpy().copy(),force=foot_forces(env).norm(dim=-1).cpu().numpy().copy(),alive=alive.copy()))
                action=policy(TensorDict(obs,batch_size=[args.envs])).clamp(-1,1);obs,_,failed,trunc,_=env.step(action)
                fresh=failed|trunc
                for name in env.termination_manager.active_terms:
                    count=int((env.termination_manager.get_term(name)&torch.as_tensor(alive,device=env.device)).sum())
                    if count:reasons[name]=reasons.get(name,0)+count
                alive&=~fresh.cpu().numpy()
                if i%250==0:progress.write_text(json.dumps(dict(state='evaluating',time_s=time,stage=stage,alive=int(alive.sum())),indent=2),encoding='utf-8')
                if not alive.any():break
        a={k:np.stack([x[k] for x in traces]) for k in traces[0]};rows=[];start=0.;passed=alive.copy()
        streak=np.zeros((args.envs,2),int);armed=np.zeros((args.envs,2),bool);landings=[]
        for low,force in zip(a['low'],a['force']):
            streak=np.where(low>.004,streak+1,0);armed|=streak>=3
            landing=armed&(force>.10*cfg.sunday_mass*9.81);armed&=~landing;landings.append(landing)
        landings=np.array(landings)
        for k,(end,vector) in enumerate(STAGES):
            ids=np.flatnonzero((a['time']>=max(start+1.2,end-4.))&(a['time']<end));row=dict(stage=k,start_s=start,end_s=end,vector=vector)
            if len(ids)<2:row.update(passed=0,not_completed=True);passed[:]=False;rows.append(row);start=end;continue
            v=a['velocity'][ids];om=a['omega'][ids];xy=a['xyz'][ids,:,:2];drift=np.linalg.norm(xy[-1]-xy[0],axis=-1)
            ok=a['alive'][ids].all(axis=0)
            if np.linalg.norm(vector[:2])==0:
                speed=np.percentile(np.linalg.norm(v[:,:,:2],axis=-1),95,axis=0);wz=np.percentile(np.abs(om[:,:,2]),95,axis=0)
                ok&=(speed<.025)&(wz<.10)&(drift<.04)
                row.update(speed_p95_mean=float(speed.mean()),drift_mean_m=float(drift.mean()))
            else:
                error=np.abs(a['heading_error'][ids]);heading95=np.percentile(error,95,axis=0)
                rmse=np.sqrt(((v[:,:,0]-a['command'][ids,:,0])**2).mean(axis=0));steps=landings[ids].sum(axis=0)
                desired=np.array(vector[:2]);desired/=np.linalg.norm(desired)
                progress_m=((xy[-1]-xy[0])*desired).sum(axis=-1)
                ok&=(heading95<.25)&(rmse<.04)&(steps.min(axis=1)>=1)&(progress_m>.03)
                row.update(heading_error_p95_mean_rad=float(heading95.mean()),forward_rmse_mean=float(rmse.mean()),landings_mean=steps.mean(axis=0).tolist(),progress_along_arrow_mean_m=float(progress_m.mean()))
            passed&=ok;row['passed']=int(ok.sum());rows.append(row);start=end
        result=dict(checkpoint=str(args.checkpoint),episodes=args.envs,survived=int(alive.sum()),passed=int(passed.sum()),
                    accepted=bool(alive.mean()>=.95 and passed.mean()>=.90),termination_reasons=reasons,stages=rows,
                    scope='Exact web mapper; speed change, stop,90deg steering,180deg reversal,release-and-hold; real PPO feedback, fixed physical model')
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');np.savez_compressed(args.output.with_suffix('.npz'),**a)
        progress.write_text(json.dumps(dict(state='completed',passed=result['accepted']),indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':main()
