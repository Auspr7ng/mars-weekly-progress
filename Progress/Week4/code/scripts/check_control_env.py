"""Exercise the actual command manager, reset, gate and turn reward paths on GPU."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse,json,torch
from mjlab.envs import ManagerBasedRlEnv
from sunday_locomotion.env_cfg import make_env
from sunday_locomotion.control_mdp import foot_targets

torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--revision',type=int,choices=[12,13,14,15,16,17,18,19,20,21,22,23,24,25],default=12)
p.add_argument('--training-starts',action='store_true');args=p.parse_args()
if args.training_starts and args.revision<21:raise ValueError('Training-start mixture starts at R21')
cfg=make_env(32,'control',True,play=not args.training_starts,revision=args.revision,nominal_start_fraction=.25 if args.training_starts else 0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    if args.revision>=16:
        from sunday_locomotion.quality_mdp import clearance_shortfall_cost
        heights=torch.arange(0,16,device=env.device)[:,None]*.001
        costs=clearance_shortfall_cost(heights,torch.ones(16,device=env.device),args.revision)[:,0]
        assert torch.all(costs[:-1]>costs[1:]) and costs[-1]<1e-10,'Lift reward must distinguish every millimetre from ground to15mm'
    if args.revision>=18:
        from sunday_locomotion.quality_mdp import load_transfer_cost
        forces=torch.tensor([[.5,.5],[.25,.75],[0.,1.]],device=env.device)
        costs=load_transfer_cost(forces,torch.tensor([[False,True]]*3,device=env.device))
        assert torch.all(costs[:-1]>costs[1:]) and costs[-1]==0,'Unloading must improve the scheduled support cost before lift-off'
    obs,_=env.reset();term=env.command_manager.get_term('twist');nominal_count=None
    if args.training_starts:
        mask=env._nominal_start_mask.clone();nominal_count=int(mask.sum());data=env.scene['robot'].data
        assert 0<nominal_count<32
        assert torch.allclose(data.joint_pos[mask],data.default_joint_pos[mask],atol=1e-6)
        assert ((data.joint_pos[~mask]-data.default_joint_pos[~mask]).abs().amax(dim=-1)>1e-6).all()
        evaluation_cfg=make_env(32,'control',True,revision=args.revision)
        assert evaluation_cfg.sunday_nominal_start_fraction==0 and 'nominal_subset' not in evaluation_cfg.events
    assert obs['actor'].shape==(32,460) and obs['critic'].shape==(32,98)
    assert torch.count_nonzero(term.value)==0 and torch.count_nonzero(term.target_yaw)==0
    if args.revision>=13:assert torch.count_nonzero(obs['actor'][:,425:435])==0
    action=torch.zeros(32,22,device=env.device);failures=0
    for target in ([.06,0,.1],[0,0,0]):
        term.override=target
        for _ in range(75):
            obs,r,done,_,_=env.step(action);failures+=int(done.sum())
            assert all(torch.isfinite(x).all() for x in obs.values()) and torch.isfinite(r).all()
        if target[2]:
            assert torch.allclose(term.value,torch.tensor(target,device=env.device).expand_as(term.value),atol=1e-5)
            assert term.target_yaw.min()>.08 and term.motion_gate.min()>.95
    assert term.value.abs().max()<1e-6 and term.motion_gate.max()<.002
    assert foot_targets(env)[4].abs().max()<.00004
    if args.revision>=15:
        term.override=[0,0,.12]
        for _ in range(75):
            obs,r,done,_,_=env.step(action);failures+=int(done.sum())
            assert torch.isfinite(r).all()
        assert term.sway_amplitude.min()>.0275
        assert term.sway_amplitude_dot.abs().max()<.003
    env.common_step_counter=16000;term.override=None;term.command_counter[:]=1
    sampled=[]
    for _ in range(20):
        term._resample_command(torch.arange(32,device=env.device));sampled.append(term.target.clone())
    commands=torch.cat(sampled)
    if args.revision>=22:
        assert commands[:,0].min()>=-.06001 and commands[:,0].min()<-.04
        assert commands[:,0].max()<=.15001 and commands[:,0].max()>.13
        assert commands[:,2].abs().max()<=.40001 and commands[:,2].abs().max()>.35
    else:assert commands[:,0].max()<=.08501 and commands[:,2].abs().max()<=.18001
    minimum_turns=10 if args.revision>=13 else 25
    assert (commands.abs().sum(dim=1)==0).sum()>60 and (commands[:,2]>.03).sum()>minimum_turns and (commands[:,2]<-.03).sum()>minimum_turns
    env.reset();assert term.value.abs().max()==0 and term.motion_gate.max()==0 and term.target_yaw.abs().max()==0
    if args.revision>=15:assert torch.allclose(term.sway_amplitude,torch.full_like(term.sway_amplitude,.018))
    assert failures==0
    report=dict(revision=args.revision,training_start_mixture=args.training_starts,initial_nominal_starts=nominal_count,evaluation_reset_mixture_disabled=True,actor_dim=460,critic_dim=98,actions=22,zero_action_failures=failures,
                sampled_commands=commands.shape[0],stopped_samples=int((commands.abs().sum(dim=1)==0).sum()),
                left_turn_samples=int((commands[:,2]>.03).sum()),right_turn_samples=int((commands[:,2]<-.03).sum()),
                max_speed=float(commands[:,0].max()),max_abs_yaw_rate=float(commands[:,2].abs().max()),passed=True)
    (WEEK/(f'results/validation/control_command_checks_r{args.revision}.json' if args.revision>=13 else 'results/validation/control_command_checks.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
finally:env.close()
