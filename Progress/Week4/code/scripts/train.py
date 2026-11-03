"""Explicit, bounded PPO launcher. No package installation or remote logging."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from sunday_locomotion.env_cfg import make_env,agent_cfg


class TrainingPaused(Exception):
    pass


class LocalRunner(MjlabOnPolicyRunner):
    def save(self,path,infos=None):
        super().save(path,infos)
        record={'state':'running','pid':os.getpid(),'checkpoint':str(Path(path).resolve()),
                'iteration':self.current_learning_iteration,
                'environment_transitions':(self.current_learning_iteration+1)*self.env.num_envs*self.cfg['num_steps_per_env'],
                'time':datetime.now(ZoneInfo('America/Indianapolis')).isoformat()}
        if Path(path).name=='model_initial.pt':record.update(iteration=-1,environment_transitions=0)
        status_path=Path(path).parent/'status.json'
        tmp=status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record,indent=2),encoding='utf-8');tmp.replace(status_path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--stage',choices=['stand','forward','walk','omni','control'],default='stand')
    p.add_argument('--whole-body',action='store_true')
    p.add_argument('--phase',action='store_true')
    p.add_argument('--revision',type=int,choices=list(range(1,26)),default=1)
    p.add_argument('--resume',type=Path)
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--initialize-from',type=Path)
    p.add_argument('--dynamics-transfer',type=Path,help='Explicit R19 weights-only adaptation from archived R17/R18 dynamics')
    p.add_argument('--zero-mean',action='store_true',help='Initialize the fresh actor mean at the verified PD reference')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.revision>=2:args.phase=True
    if args.dynamics_transfer and (args.revision!=19 or args.resume or args.initialize_from or args.warm_start or args.zero_mean):
        raise ValueError('Dynamics transfer is an exclusive R19 weights-only initialization')
    if args.resume and args.zero_mean:raise ValueError('zero-mean is only valid for a fresh actor')
    if args.warm_start and (args.revision!=4 or args.resume or args.zero_mean):raise ValueError('warm-start requires revision 4 and cannot combine with resume/zero-mean')
    if args.initialize_from and (args.revision not in (5,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25) or args.resume or args.zero_mean or args.warm_start):raise ValueError('initialize-from requires a supported explicit weight transfer')
    if args.stage=='control' and (args.revision<12 or not args.whole_body):raise ValueError('Control curriculum requires R12 full-body policy')
    if args.iterations<1 or args.iterations>10000:raise ValueError('iterations must be explicitly bounded to 1..10000')
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    cfg=make_env(args.envs,args.stage,args.whole_body,gait_phase=args.phase,revision=args.revision,nominal_start_fraction=.25 if args.revision>=21 else 0.);cfg.seed=args.seed
    if args.revision==24:cfg.sunday_action_gain_curriculum=True
    if args.revision>=25:cfg.sunday_cadence_curriculum=not bool(args.resume)
    acfg=agent_cfg(args.revision);acfg.seed=args.seed;acfg.max_iterations=args.iterations
    # Save the exact Python training source beside each run as well as config.
    source_files=[Path(__file__),WEEK/'code/src/sunday_locomotion/mdp.py',WEEK/'code/src/sunday_locomotion/env_cfg.py',WEEK/'code/src/sunday_locomotion/warmstart.py',WEEK/'code/src/sunday_locomotion/quality_mdp.py',WEEK/'code/src/sunday_locomotion/control_mdp.py']
    if args.revision>=17:source_files.extend([WEEK/'code/src/sunday_locomotion/symmetry.py',WEEK/'code/src/sunday_locomotion/learning_cfg.py',WEEK/'config/mirror_map.json'])
    if args.revision>=20:source_files.append(WEEK/'code/src/sunday_locomotion/clearance_mdp.py')
    if args.revision>=23:source_files.append(WEEK/'code/src/sunday_locomotion/action_ranges.py')
    if args.revision>=25:source_files.append(WEEK/'code/src/sunday_locomotion/cadence.py')
    for file in source_files:
        (run/'source').mkdir(exist_ok=True)
        (run/'source'/file.name).write_bytes(file.read_bytes())
    (run/'dynamics').mkdir(exist_ok=True)
    for relative in ('model/sunday_a1_locomotion.xml','model/home.json','config/hardware_confirmation.json',
                     'config/dynamics_profile.json','config/simulation_assumptions.json'):
        source=WEEK/relative
        (run/'dynamics'/source.name).write_bytes(source.read_bytes())
    nominal=WEEK/('model/arm_clearance_home.json' if args.revision>=24 else ('model/arm_home.json' if args.revision>=7 else ('model/natural_home.json' if args.revision>=6 else ('model/quality_home.json' if args.revision>=5 else ('model/gait_home.json' if args.revision>=2 else 'model/home.json')))))
    (run/'dynamics'/('arm_home.json' if args.revision>=24 else nominal.name)).write_bytes(nominal.read_bytes())
    spec=dict(stage=args.stage,revision=args.revision,nominal_start_fraction=cfg.sunday_nominal_start_fraction,nominal_sha256=hashlib.sha256(nominal.read_bytes()).hexdigest(),whole_body=args.whole_body,gait_phase=args.phase,num_envs=args.envs,iterations=args.iterations,seed=args.seed,
              resume=str(args.resume.resolve()) if args.resume else None,warm_start=str(args.warm_start.resolve()) if args.warm_start else None,initialize_from=str(args.initialize_from.resolve()) if args.initialize_from else None,dynamics_transfer=str(args.dynamics_transfer.resolve()) if args.dynamics_transfer else None,zero_mean_initialization=args.zero_mean,simulation_only=True,hardware_calibrated=False,
              actor_linear_velocity_source='simulated base velocity' if args.revision>=10 else None,real_velocity_estimator_required=args.revision>=10,
              model_sha256=hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest(),
              source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (run/'source').iterdir()},
              dependencies={n:importlib.metadata.version(n) for n in ['mjlab','mujoco','mujoco-warp','warp-lang','torch','rsl-rl-lib']},
              started=datetime.now(ZoneInfo('America/Indianapolis')).isoformat())
    (run/'run.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    dump_yaml(run/'env.yaml',asdict(cfg));dump_yaml(run/'agent.yaml',asdict(acfg))
    record={'state':'initializing','pid':os.getpid(),'run_dir':str(run),'stage':args.stage}
    (run/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    env=None
    try:
        env=ManagerBasedRlEnv(cfg,device='cuda:0')
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
        runner=LocalRunner(wrapper,asdict(acfg),str(run),device='cuda:0')
        if args.dynamics_transfer:
            from sunday_locomotion.warmstart import load_metadata_dynamics_transfer
            validation=load_metadata_dynamics_transfer(runner,args.dynamics_transfer,spec['model_sha256'],spec['nominal_sha256'])
            (run/'initialization.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
            runner.save(str(run/'model_initial.pt'))
        elif args.initialize_from and args.revision>=24 and json.loads((args.initialize_from.parent/'run.json').read_text(encoding='utf-8'))['nominal_sha256']!=spec['nominal_sha256']:
            from sunday_locomotion.warmstart import load_arm_clearance_transfer
            validation=load_arm_clearance_transfer(runner,args.initialize_from,spec['model_sha256'],spec['nominal_sha256'],target_revision=args.revision)
            (run/'initialization.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
            runner.save(str(run/'model_initial.pt'))
        elif args.initialize_from and args.revision>=12:
            from sunday_locomotion.warmstart import load_command_warm_start
            validation=load_command_warm_start(runner,args.initialize_from,spec['model_sha256'],spec['nominal_sha256'],args.revision)
            (run/'initialization.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
            runner.save(str(run/'model_initial.pt'))
        elif args.initialize_from and args.revision==10:
            from sunday_locomotion.warmstart import load_velocity_warm_start
            validation=load_velocity_warm_start(runner,args.initialize_from,spec['model_sha256'],spec['nominal_sha256'])
            (run/'initialization.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
            runner.save(str(run/'model_initial.pt'))
        elif args.initialize_from:
            prior=json.loads((args.initialize_from.parent/'run.json').read_text())
            expected_revision=10 if args.revision==11 else (8 if args.revision==9 else (7 if args.revision==8 else 4))
            if (prior.get('revision')!=expected_revision or prior['whole_body']!=args.whole_body or prior['model_sha256']!=spec['model_sha256'] or
                (args.revision>=8 and prior['nominal_sha256']!=spec['nominal_sha256'])):
                raise ValueError('Quality transfer requires the expected prior task and identical physical model/action schema')
            weights=torch.load(args.initialize_from,map_location='cuda:0',weights_only=False)
            runner.alg.actor.load_state_dict(weights['actor_state_dict'])
            runner.alg.critic.load_state_dict(weights['critic_state_dict'])
            with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(min=.08,max=.12 if args.revision>=8 else 1.)
            (run/'initialization.json').write_text(json.dumps(dict(source=str(args.initialize_from.resolve()),optimizer_reset=True,
                changed_nominal_pose=args.revision==5,changed_phase_period=args.revision==5,source_revision=prior['revision'],
                same_physical_mass_inertia_torque_limits=True),indent=2),encoding='utf-8')
            runner.save(str(run/'model_initial.pt'))
        if args.warm_start:
            from sunday_locomotion.warmstart import load_heading_warm_start
            validation=load_heading_warm_start(runner,args.warm_start,spec['model_sha256'],spec['nominal_sha256'])
            (run/'warm_start_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
            runner.save(str(run/'model_initial.pt'))
        if args.zero_mean:
            output=[layer for layer in runner.alg.get_policy().mlp.modules() if isinstance(layer,torch.nn.Linear)][-1]
            with torch.no_grad():output.weight.zero_();output.bias.zero_()
            with torch.inference_mode():
                initial=runner.get_inference_policy(device='cuda:0')(wrapper.get_observations())
                assert torch.count_nonzero(initial)==0,'Initial deterministic actor must match the PD reference'
            runner.alg.train_mode()
        original_log=runner.logger.log
        def log_with_stop(*a,**kw):
            original_log(*a,**kw)
            if (run/'STOP').exists():
                runner.save(str(run/f'model_{runner.current_learning_iteration}_paused.pt'))
                raise TrainingPaused('Requested by STOP file')
        runner.logger.log=log_with_stop
        if args.resume:
            prior=json.loads((args.resume.parent/'run.json').read_text())
            if prior['whole_body']!=args.whole_body or prior['gait_phase']!=args.phase:
                raise ValueError('Checkpoint action/observation schema differs')
            if prior['model_sha256']!=spec['model_sha256']:raise ValueError('Checkpoint model hash differs')
            if prior.get('revision',1)!=args.revision or prior.get('nominal_sha256',spec['nominal_sha256'])!=spec['nominal_sha256']:
                raise ValueError('Checkpoint task revision or nominal pose differs')
            runner.load(str(args.resume))
        start=time.perf_counter()
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        result=json.loads((run/'status.json').read_text())
        result.update(state='completed',training_elapsed_s=time.perf_counter()-start,
                      gpu_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2)
        (run/'status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('TRAINING_COMPLETED '+json.dumps(result),flush=True)
    except TrainingPaused:
        result=json.loads((run/'status.json').read_text(encoding='utf-8'))
        result.update(state='paused',reason='STOP file')
        (run/'status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('TRAINING_PAUSED '+json.dumps(result),flush=True)
    except BaseException as exc:
        failure={'state':'failed','error':repr(exc),'pid':os.getpid()}
        (run/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        (run/'status.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
