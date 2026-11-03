"""Audit stance/impact/oscillation quality and native closed-loop transfer."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse,hashlib,json,math
from dataclasses import asdict
from pathlib import Path
import mujoco,numpy as np,torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from sunday_locomotion.env_cfg import make_env,agent_cfg
from sunday_locomotion.mdp import foot_forces
from sunday_locomotion.gait_metrics import detailed_metrics


from sunday_locomotion.native_observations import NativeHistory


def observe(m,d,force,peak,torque):
    root=m.body('robot/base_link').id
    r=d.xmat[root].reshape(3,3)
    euler=np.array([math.atan2(r[2,1],r[2,2]),math.asin(np.clip(-r[2,0],-1,1)),math.atan2(r[1,0],r[0,0])])
    sites=[m.site('robot/'+s+'_sole').id for s in ('l','r')]
    rotation=d.site_xmat[sites].reshape(2,3,3)
    low=d.site_xpos[sites,2]-np.abs(rotation[:,2,0])*.0525-np.abs(rotation[:,2,1])*.09
    vel=[]
    for sid in sites:
        v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,sid,v,0);vel.append(v[3:])
    return dict(xyz=d.qpos[:3].copy(),rpy=euler,sole_low=low,sole_pos=d.site_xpos[sites].copy(),
                sole_vel=np.asarray(vel),foot_force=np.asarray(force),peak_force=np.asarray(peak),torque=np.asarray(torque).copy())


def summarize(trace,mass):
    a={k:np.asarray([t[k] for t in trace]) for k in trace[0]}
    warm=slice(min(100,len(trace)//4),None)
    force=a['foot_force'][warm];support=force>.1*mass*9.81
    slip=np.linalg.norm(a['sole_vel'][warm,:,:2],axis=-1)
    flight=(~support).all(axis=1)
    z=a['xyz'][warm,2];rpy=a['rpy'][warm]
    report=dict(duration_s=len(trace)*.02,forward_m=float(a['xyz'][-1,0]),
                height_peak_to_peak_mm=float(np.ptp(z)*1000),
                roll_peak_to_peak_deg=float(np.ptp(rpy[:,0])*180/math.pi),
                pitch_peak_to_peak_deg=float(np.ptp(rpy[:,1])*180/math.pi),
                supported_sole_speed_mean_m_s=float(slip[support].mean()) if support.any() else None,
                supported_sole_speed_p95_m_s=float(np.percentile(slip[support],95)) if support.any() else None,
                peak_foot_force_bodyweights=float(a['peak_force'].max()/(mass*9.81)),
                flight_fraction=float(flight.mean()),
                left_right_support_fraction=support.mean(axis=0).tolist(),
                foot_height_max_mm=(a['sole_low'][warm].max(axis=0)*1000).tolist(),
                foot_below_ground_min_mm=(a['sole_low'][warm].min(axis=0)*1000).tolist())
    return report,a


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--tag')
    p.add_argument('--speed',type=float,default=.08)
    args=p.parse_args();spec=json.loads((args.checkpoint.parent/'run.json').read_text())
    revision=spec['revision']
    if revision not in (4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25):raise ValueError('Unsupported actor schema')
    if hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()!=spec['model_sha256']:
        raise ValueError('Current physical model does not match this checkpoint')
    tag=args.tag or f'r{revision}'
    progress=WEEK/f'results/quality/{tag}_progress.json'
    def mark(state,**kw):progress.write_text(json.dumps(dict(state=state,checkpoint=str(args.checkpoint.resolve()),**kw),indent=2),encoding='utf-8')
    mark('initializing')
    torch.set_num_threads(4)
    nominal=args.checkpoint.parent/'dynamics'/('arm_home.json' if revision>=7 else ('natural_home.json' if revision>=6 else ('quality_home.json' if revision>=5 else 'gait_home.json')))
    if hashlib.sha256(nominal.read_bytes()).hexdigest()!=spec['nominal_sha256']:raise ValueError('Archived nominal pose hash mismatch')
    cfg=make_env(1,spec['stage'],spec['whole_body'],play=True,revision=revision,nominal_path=nominal);cfg.auto_reset=False;cfg.rewards={}
    env=ManagerBasedRlEnv(cfg,device='cuda:0');m=env.sim.mj_model
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(revision)),log_dir=None,device='cuda:0');runner.load(str(args.checkpoint))
        policy=runner.get_inference_policy(device='cuda:0')
        print('AUDIT_POLICY_LOADED '+str(args.checkpoint),flush=True)
        mark('gpu_rollout')
        env.command_manager.get_term('twist').override=[args.speed,0,0]
        obs,_=env.reset(seed=741)
        initial_q=env.sim.data.qpos[0].cpu().numpy().copy();initial_v=env.sim.data.qvel[0].cpu().numpy().copy()
        default=env.scene['robot'].data.default_joint_pos[0].cpu().numpy().copy()
        names=list(env.scene['robot'].joint_names);action=env.action_manager.get_term('residual')
        initial_command=env.command_manager.get_command('twist')[0].cpu().numpy().copy()
        actdim=len(action.names)
        compare=mujoco.MjData(m);builder=NativeHistory(m,default,names,actdim,revision)
        traces=[];max_error=0.;term_errors=np.zeros(9 if revision>=10 else 8);gpu_reasons=[]
        with torch.inference_mode():
            for step in range(round(args.seconds/.02)):
                compare.qpos[:]=env.sim.data.qpos[0].cpu().numpy();compare.qvel[:]=env.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,compare)
                own=builder.push(compare,env.command_manager.get_command('twist')[0].cpu().numpy(),env.action_manager.action[0].cpu().numpy(),step)
                max_error=max(max_error,float(np.max(np.abs(own-obs['actor'][0].cpu().numpy()))))
                split_sizes=[15,15,15,135,135,actdim*5,10]+([10] if revision>=10 else [])
                chunks=np.split(np.abs(own-obs['actor'][0].cpu().numpy()),np.cumsum(split_sizes))
                term_errors=np.maximum(term_errors,[x.max() for x in chunks])
                if max(term_errors)>1e-4:raise AssertionError(f'Native observation mapping mismatch: {term_errors}')
                out=policy(TensorDict(obs,batch_size=[1])).clamp(-1,1)
                obs,_,term,trunc,_=env.step(out)
                compare.qpos[:]=env.sim.data.qpos[0].cpu().numpy();compare.qvel[:]=env.sim.data.qvel[0].cpu().numpy();mujoco.mj_forward(m,compare)
                f=torch.linalg.vector_norm(foot_forces(env),dim=-1)[0].cpu().numpy()
                sensor=env.scene['feet_contact'];order=[sensor.primary_names.index('l_foot_link_1'),sensor.primary_names.index('r_foot_link_1')]
                peak=torch.linalg.vector_norm(sensor.data.force_history[:,order],dim=-1).max(dim=-1).values[0].cpu().numpy()
                sample=observe(m,compare,f,peak,env.scene['robot'].data.actuator_force[0].cpu().numpy())
                if revision>=25:sample['gait_phase']=float(env.command_manager.get_term('twist').gait_phase[0])
                traces.append(sample)
                if term.any() or trunc.any():
                    gpu_reasons=[name for name in env.termination_manager.active_terms if env.termination_manager.get_term(name).any()]
                    break
        gpu,a=summarize(traces,cfg.sunday_mass)
        metric_args=dict(period=1.4 if revision>=5 else 1.,offset=.075 if revision>=5 else 0.,
                         actuator_names=[m.actuator(i).name.removeprefix('robot/') for i in range(m.nu)],force_limits=np.abs(m.actuator_forcerange).max(axis=1))
        gpu.update(detailed_metrics(a,cfg.sunday_mass,**metric_args))
        print('AUDIT_GPU_COMPLETE '+json.dumps(gpu),flush=True)
        mark('native_rollout',gpu=gpu)
        gpu['termination_reasons']=gpu_reasons
        output=WEEK/'results/quality';output.mkdir(exist_ok=True)
        np.savez_compressed(output/f'{tag}_gpu_trace.npz',**a)
        d=mujoco.MjData(m);d.qpos[:]=initial_q;d.qvel[:]=initial_v;mujoco.mj_forward(m,d)
        builder=NativeHistory(m,default,names,actdim,revision)
        command=initial_command.copy()
        last=np.zeros(actdim);targets=default.copy();joint_ids=action.ids.cpu().numpy();scales=action.scales.cpu().numpy();speeds=action.speeds.cpu().numpy()
        limits=env.scene['robot'].data.soft_joint_pos_limits[0].cpu().numpy()
        native=[];failed=False;native_self_contacts=set()
        feet_body=[m.body('robot/'+s+'_foot_link_1').id for s in ('l','r')]
        with torch.inference_mode():
            for step in range(round(args.seconds/.02)):
                ob=builder.push(d,command,last,step)
                out=policy(TensorDict({'actor':torch.as_tensor(ob[None],device='cuda:0')},batch_size=[1]))[0].clamp(-1,1).cpu().numpy()
                from sunday_locomotion.action_ranges import native_scales
                requested=default[joint_ids]+out*native_scales(scales,action.names,command,revision)
                requested=targets[joint_ids]+.3*(requested-targets[joint_ids])
                requested=np.clip(requested,targets[joint_ids]-speeds*.02,targets[joint_ids]+speeds*.02)
                targets[joint_ids]=np.clip(requested,limits[joint_ids,0],limits[joint_ids,1]);last=out
                d.ctrl[:]=[targets[names.index(m.actuator(i).name.removeprefix('robot/'))] for i in range(m.nu)]
                subforces=[]
                for _ in range(10):
                    mujoco.mj_step(m,d);f=np.zeros(2)
                    for c in range(d.ncon):
                        contact=d.contact[c];b1=int(m.geom_bodyid[contact.geom1]);b2=int(m.geom_bodyid[contact.geom2])
                        for j,bid in enumerate(feet_body):
                            if bid in (b1,b2) and (m.geom_type[contact.geom1]==mujoco.mjtGeom.mjGEOM_PLANE or m.geom_type[contact.geom2]==mujoco.mjtGeom.mjGEOM_PLANE):
                                force=np.zeros(6);mujoco.mj_contactForce(m,d,c,force);f[j]+=abs(force[0])
                    subforces.append(f)
                mujoco.mj_forward(m,d)
                for c in d.contact:
                    if c.dist<0 and m.geom(c.geom1).name.startswith('robot/') and m.geom(c.geom2).name.startswith('robot/'):
                        native_self_contacts.add((m.geom(c.geom1).name,m.geom(c.geom2).name))
                sample=observe(m,d,np.mean(subforces,axis=0),np.max(subforces,axis=0),d.actuator_force)
                if revision>=22:command[0]+=np.clip(args.speed-command[0],-.01,.01)
                else:command[0]=min(args.speed,command[0]+(.005 if revision>=12 else .01))
                if revision>=25:sample['gait_phase']=float(builder.next_phase(command))
                native.append(sample)
                if d.qpos[2]<.15 or abs(native[-1]['rpy'][0])>1 or abs(native[-1]['rpy'][1])>1:failed=True;break
        cpu,b=summarize(native,cfg.sunday_mass);cpu['failed']=failed
        cpu.update(detailed_metrics(b,cfg.sunday_mass,**metric_args))
        cpu['self_contact_pairs']=sorted(native_self_contacts)
        np.savez_compressed(output/f'{tag}_native_trace.npz',**b)
        report=dict(checkpoint=str(args.checkpoint.resolve()),revision=revision,command_m_s=args.speed,model_sha256=spec['model_sha256'],
                    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    nominal_sha256=spec['nominal_sha256'],gpu=gpu,native=cpu,native_observation_max_error=max_error,
                    native_observation_errors=dict(zip(['gyro','gravity','command','joint_pos','joint_vel','action','phase','heading']+(['velocity_feedback'] if revision>=10 else []),term_errors.tolist())),
                    notes='Native rollout uses its own physics and feedback after a single reset. No root driving or copying GPU poses into native rollout.')
        (output/f'{tag}_quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        mark('completed',quality_report=str(output/f'{tag}_quality.json'))
        print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':main()
