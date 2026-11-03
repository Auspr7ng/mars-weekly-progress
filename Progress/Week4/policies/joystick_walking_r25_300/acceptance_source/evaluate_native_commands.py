"""Independent native MuJoCo feedback rollouts of the command programs."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse,hashlib,json,math
from pathlib import Path
from dataclasses import asdict
import mujoco,numpy as np,torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from sunday_locomotion.env_cfg import make_env,agent_cfg
from sunday_locomotion.command_programs import PROGRAMS,command_at,duration
from sunday_locomotion.control_metrics import summarize
from audit_gait_quality import NativeHistory,observe


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--programs',nargs='+',choices=list(PROGRAMS),default=['idle','speed_stop','turns','in_place'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--actor-device',choices=['cuda:0','cpu'],default='cuda:0');args=p.parse_args();run=args.checkpoint.parent
    spec=json.loads((run/'run.json').read_text());revision=spec['revision'];nominal=run/'dynamics/arm_home.json'
    if revision<12:raise ValueError('Native transition runner currently requires R12')
    if hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()!=spec['model_sha256']:raise ValueError('Physical model differs')
    if hashlib.sha256(nominal.read_bytes()).hexdigest()!=spec['nominal_sha256']:raise ValueError('Nominal pose differs')
    torch.set_num_threads(4)
    env=ManagerBasedRlEnv(make_env(1,spec['stage'],True,play=True,revision=revision,nominal_path=nominal),device='cuda:0')
    args.output.parent.mkdir(parents=True,exist_ok=True);reports=[]
    try:
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=MjlabOnPolicyRunner(wrapper,asdict(agent_cfg(revision)),log_dir=None,device='cuda:0');runner.load(str(args.checkpoint))
        policy=runner.get_inference_policy(device=args.actor_device);m=env.sim.mj_model
        names=list(env.scene['robot'].joint_names);action=env.action_manager.get_term('residual')
        ids=action.ids.cpu().numpy();scales=action.scales.cpu().numpy();speeds=action.speeds.cpu().numpy()
        default=env.scene['robot'].data.default_joint_pos[0].cpu().numpy().copy()
        limits=env.scene['robot'].data.soft_joint_pos_limits[0].cpu().numpy()
        actuator_joints=[names.index(m.actuator(i).name.removeprefix('robot/')) for i in range(m.nu)]
        foot_bodies=[m.body('robot/'+s+'_foot_link_1').id for s in ('l','r')]
        mass=env.cfg.sunday_mass
        for program in args.programs:
            env.command_manager.get_term('twist').override=command_at(program,0)
            with torch.inference_mode():obs,_=env.reset(seed=741)
            d=mujoco.MjData(m);d.qpos[:]=env.sim.data.qpos[0].cpu().numpy();d.qvel[:]=env.sim.data.qvel[0].cpu().numpy();d.ctrl[:]=default[actuator_joints]
            mujoco.mj_forward(m,d)
            builder=NativeHistory(m,default,names,len(ids),revision)
            command=np.zeros(3);heading=0.;last=np.zeros(len(ids));targets=default.copy()
            mapped=builder.push(d,command,last,0,heading)
            mapping_error=float(np.max(np.abs(mapped-obs['actor'][0].cpu().numpy())))
            if mapping_error>1e-4:raise AssertionError('Initial native observation differs: '+str(mapping_error))
            # Reinitialize history: the first actor call below must not advance it twice.
            builder=NativeHistory(m,default,names,len(ids),revision)
            force=np.zeros(2);peak=np.zeros(2);trace=[];previous_yaw=0.;unwrapped=0.;failed=False;reasons=[]
            counters=dict(tilt=0,height=0,ground=0,self=0);contact_pairs={}
            with torch.inference_mode():
                for step in range(round(duration(program)/.02)):
                    data=observe(m,d,force,peak,d.actuator_force)
                    velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,m.body('robot/base_link').id,velocity,1)
                    yaw=float(data['rpy'][2]);delta=yaw-previous_yaw;unwrapped+=math.atan2(math.sin(delta),math.cos(delta));previous_yaw=yaw
                    err=heading-yaw
                    trace.append(dict(time=step*.02,xyz=data['xyz'],velocity=velocity[3:],omega=velocity[:3],command=command.copy(),
                                      heading_error=math.atan2(math.sin(err),math.cos(err)),yaw_unwrapped=unwrapped,
                                      sole_low=data['sole_low'],foot_force=force.copy(),foot_speed=np.linalg.norm(data['sole_vel'][:,:2],axis=-1),alive=True))
                    observation=builder.push(d,command,last,step,heading)
                    out=policy(TensorDict({'actor':torch.as_tensor(observation[None],device=args.actor_device)},batch_size=[1]))[0].clamp(-1,1).cpu().numpy()
                    from sunday_locomotion.action_ranges import native_scales
                    request=default[ids]+out*native_scales(scales,action.names,command,revision);request=targets[ids]+.3*(request-targets[ids])
                    request=np.clip(request,targets[ids]-speeds*.02,targets[ids]+speeds*.02)
                    targets[ids]=np.clip(request,limits[ids,0],limits[ids,1]);d.ctrl[:]=targets[actuator_joints];last=out
                    subforces=[];body_ground=False;self_contact=False
                    for _ in range(10):
                        mujoco.mj_step(m,d);f=np.zeros(2)
                        for ci in range(d.ncon):
                            c=d.contact[ci];b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
                            plane=m.geom_type[c.geom1]==mujoco.mjtGeom.mjGEOM_PLANE or m.geom_type[c.geom2]==mujoco.mjtGeom.mjGEOM_PLANE
                            cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf)
                            for j,b in enumerate(foot_bodies):
                                if plane and b in (b1,b2):f[j]+=abs(cf[0])
                            if plane and b1 not in foot_bodies and b2 not in foot_bodies and abs(cf[0])>.02*mass*9.81:body_ground=True
                            if not plane and c.dist<0 and m.geom(c.geom1).name.startswith('robot/') and m.geom(c.geom2).name.startswith('robot/'):
                                self_contact=True
                                pair=tuple(sorted((m.geom(c.geom1).name,m.geom(c.geom2).name)))
                                entry=contact_pairs.setdefault(pair,dict(geoms=list(pair),first_time_s=float(d.time),last_time_s=float(d.time),minimum_distance_m=0.,peak_normal_force_N=0.,contact_samples=0))
                                entry.update(last_time_s=float(d.time),minimum_distance_m=min(entry['minimum_distance_m'],float(c.dist)),peak_normal_force_N=max(entry['peak_normal_force_N'],abs(float(cf[0]))),contact_samples=entry['contact_samples']+1)
                        subforces.append(f)
                    mujoco.mj_forward(m,d);force=np.mean(subforces,axis=0);peak=np.max(subforces,axis=0)
                    r=d.xmat[m.body('robot/base_link').id].reshape(3,3);cos=float(r[2,2])
                    bad=dict(tilt=cos<.5,height=d.qpos[2]<.55*env.cfg.sunday_h0,ground=body_ground,self=self_contact)
                    for k,v in bad.items():counters[k]=counters[k]+1 if v else 0
                    reasons=[k for k in counters if counters[k]>=(5 if k in ('height','tilt') else 2)]
                    if cos<.173648:reasons.append('extreme_tilt')
                    if reasons:failed=True;break
                    target=np.asarray(command_at(program,step*.02))
                    from sunday_locomotion.control_mdp import slew_rates
                    rates=np.array(slew_rates(revision))*.02
                    command+=np.clip(target-command,-rates,rates)
                    heading+=command[2]*.02
            a={k:np.asarray([x[k] for x in trace]) for k in trace[0]}
            for key in a:
                if key!='time':a[key]=np.expand_dims(a[key],1)
            if failed:a['alive'][-1]=False
            report=summarize(a,program,mass);report.update(termination_reasons=reasons,initial_observation_max_error=mapping_error,planned_seconds=duration(program),self_contact_pairs=list(contact_pairs.values()))
            reports.append(report);np.savez_compressed(args.output.with_name(args.output.stem+'_'+program+'_trace.npz'),**a)
            print('NATIVE_PROGRAM_RESULT '+json.dumps(report),flush=True)
        result=dict(checkpoint=str(args.checkpoint.resolve()),actor_device=args.actor_device,programs=reports,all_requested_programs_passed=all(x['engineering_passed'] for x in reports),
                    notes='One native CPU MuJoCo feedback rollout per program, actor inference on GPU. After reset it uses only its own physical states; no copying of GPU rollout poses, no root actuation.')
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:env.close()


if __name__=='__main__':main()
