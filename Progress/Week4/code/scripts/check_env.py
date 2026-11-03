import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse,json,time,hashlib
import torch
from mjlab.envs import ManagerBasedRlEnv
from sunday_locomotion.env_cfg import make_env


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--envs',type=int,default=16)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--revision',type=int,choices=list(range(1,20)),default=1)
    parser.add_argument('--whole-body',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    env=ManagerBasedRlEnv(make_env(args.envs,whole_body=args.whole_body,play=True,revision=args.revision),device='cuda:0')
    obs,_=env.reset()
    actor_size=(395 if args.revision>=4 else (385 if args.revision>=2 else 375))+(50 if args.whole_body else 0)+(15 if args.revision>=10 else 0)
    assert obs['actor'].shape==(args.envs,actor_size)
    assert not env.scene['robot'].is_fixed_base
    assert abs(float(env.sim.mj_model.body_mass.sum())-2.84)<1e-8
    act=torch.zeros(args.envs,22 if args.whole_body else 12,device=env.device)
    failures=0;timeouts=0;start=time.perf_counter()
    force_samples=[]
    for t in range(args.steps):
        obs,r,term,trunc,extras=env.step(act)
        assert all(torch.isfinite(v).all() for v in obs.values())
        assert torch.isfinite(r).all()
        failures+=int(term.sum());timeouts+=int(trunc.sum())
        if t>20:force_samples.append(torch.linalg.vector_norm(env.scene['feet_contact'].data.force,dim=-1).sum(dim=-1).mean().item())
    data=env.scene['robot'].data
    report=dict(num_envs=args.envs,control_steps=args.steps,env_transitions=args.steps*args.envs,elapsed_s=time.perf_counter()-start,
                actor_shape=list(obs['actor'].shape),critic_shape=list(obs['critic'].shape),action_order=env.action_manager.get_term('residual').names,
                failures=failures,timeouts=timeouts,mean_root_height=float(data.root_link_pos_w[:,2].mean()),
                mean_ground_force_N=sum(force_samples)/len(force_samples),expected_weight_N=env.cfg.sunday_mass*9.81,
                gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                feet_primary_names=env.scene['feet_contact'].primary_names,
                model_sha256=hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest(),
                simulation_mass_kg=float(env.sim.mj_model.body_mass.sum()))
    assert failures==0, 'Zero-action stance failed; do not start PPO yet'
    assert abs(report['mean_ground_force_N']/report['expected_weight_N']-1)<.05
    report['revision']=args.revision
    filename=f'gpu_environment_r{args.revision}.json' if args.revision>=2 else 'gpu_environment.json'
    (WEEK/'results/validation'/filename).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CHECK_RESULT '+json.dumps(report),flush=True)
    env.close()


if __name__=='__main__':main()
