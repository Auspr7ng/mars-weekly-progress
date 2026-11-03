"""Expand revision-3 observations for heading feedback without changing initial actions."""
from pathlib import Path
import json
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def load_metadata_dynamics_transfer(runner,checkpoint,model_hash,nominal_hash):
    """Transfer weights across the explicitly recorded R19 inertial revision only."""
    import hashlib
    from .paths import WEEK,MODEL
    path=Path(checkpoint);spec=json.loads((path.parent/'run.json').read_text())
    profile=json.loads((WEEK/'config/dynamics_profile.json').read_text())
    archived=path.parent/'dynamics/sunday_a1_locomotion.xml'
    if spec['revision'] not in (17,18) or not spec['whole_body'] or not spec['gait_phase']:
        raise ValueError('Metadata adaptation requires an archived R17/R18 whole-body policy')
    if profile.get('revision')!=19 or model_hash!=profile['model_sha256'] or hashlib.sha256(MODEL.read_bytes()).hexdigest()!=model_hash:
        raise ValueError('Target must be the recorded metadata dynamics')
    if spec['model_sha256']!=profile['superseded_model_sha256'] or hashlib.sha256(archived.read_bytes()).hexdigest()!=spec['model_sha256']:
        raise ValueError('Source must use the recorded previous dynamics')
    if spec['nominal_sha256']!=nominal_hash:
        raise ValueError('This transfer preserves nominal pose and action meaning')
    saved=torch.load(path,map_location=runner.device,weights_only=False);result={}
    for name,network,dim in [('actor',runner.alg.actor,460),('critic',runner.alg.critic,98)]:
        source=saved[name+'_state_dict']
        if source['mlp.0.weight'].shape[1]!=dim:raise ValueError('Observation schema mismatch')
        network.load_state_dict(source)
        assert all(torch.equal(v,network.state_dict()[k]) for k,v in source.items())
        result[name+'_weights_exactly_copied']=True
    with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(min=.06,max=.09)
    runner.alg.train_mode()
    return dict(**result,source=str(path.resolve()),source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_model_sha256=spec['model_sha256'],target_model_sha256=model_hash,
                optimizer_reset=True,iteration_reset=True,dynamics_changed=True,changed_nominal_pose=False,
                scope='PPO adaptation on revised mass/COM/inertia; old performance is not retained as a verified result')


def load_heading_warm_start(runner,checkpoint,model_hash,nominal_hash):
    path=Path(checkpoint)
    spec=json.loads((path.parent/'run.json').read_text())
    if spec.get('revision')!=3 or spec['whole_body'] or not spec['gait_phase']:
        raise ValueError('Heading warm start supports the revision-3 leg/phase policy only')
    if spec['model_sha256']!=model_hash or spec['nominal_sha256']!=nominal_hash:
        raise ValueError('Warm start dynamics/nominal pose mismatch')
    saved=torch.load(path,map_location=runner.device,weights_only=False)
    mapping_actor=list(range(385))
    mapping_critic=list(range(77))+list(range(79,88))
    results={}
    for name,network,mapping,old_dim,new_dim,outputs in (
        ('actor',runner.alg.actor,mapping_actor,385,395,12),
        ('critic',runner.alg.critic,mapping_critic,86,88,1)):
        source=saved[name+'_state_dict']
        dest=network.state_dict()
        if dest['mlp.0.weight'].shape[1]!=new_dim or source['mlp.0.weight'].shape[1]!=old_dim:
            raise ValueError('Unexpected observation dimensions during warm start')
        for key,value in source.items():
            if dest[key].shape==value.shape:dest[key]=value.clone()
            elif key=='mlp.0.weight':
                dest[key].zero_();dest[key][:,mapping]=value
            elif key in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
                dest[key][...,mapping]=value
            else:raise ValueError(f'Unrecognized expanded parameter: {key}')
        network.load_state_dict(dest)
        # Compare both networks on varied observations. Added heading channels
        # start with zero weights; original features and normalizers are mapped.
        probe=torch.randn(64,new_dim,device=runner.device)
        probe[:,mapping]=source['obs_normalizer._mean']+source['obs_normalizer._std']*torch.randn(64,old_dim,device=runner.device)
        old_obs=TensorDict({name:probe[:,mapping]},batch_size=[64])
        distribution={'class_name':'GaussianDistribution','init_std':.22,'std_type':'scalar'} if name=='actor' else None
        old_model=MLPModel(old_obs,{name:[name]},name,outputs,hidden_dims=(256,128,128),activation='elu',obs_normalization=True,distribution_cfg=distribution).to(runner.device)
        old_model.load_state_dict(source);old_model.eval();network.eval()
        # TF32 uses different accumulation tiling when input width changes.
        # Verify the mapping in full float32, independent of training's TF32.
        tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        try:
            with torch.inference_mode():
                delta=(old_model(old_obs)-network(TensorDict({name:probe},batch_size=[64]))).abs().max().item()
        finally:torch.backends.cuda.matmul.allow_tf32=tf32
        if delta>2e-5:raise AssertionError(f'{name} changed during feature expansion: {delta}')
        results[name+'_max_output_difference']=delta
    # Small exploration for corrections, while the deterministic gait is retained.
    with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(min=.08)
    runner.alg.train_mode()
    results.update(source_checkpoint=str(path.resolve()),optimizer_reset=True,iteration_reset=True,extra_feature_weights_initialized_zero=True,comparison_precision='float32 without TF32')
    return results


def load_velocity_warm_start(runner,checkpoint,model_hash,nominal_hash):
    """Add 15 velocity-history inputs with initially zero influence on actions."""
    path=Path(checkpoint);spec=json.loads((path.parent/'run.json').read_text())
    if spec.get('revision') not in (8,9) or not spec['whole_body'] or not spec['gait_phase']:
        raise ValueError('Velocity transfer requires the 22-action revision-8/9 policy')
    if spec['model_sha256']!=model_hash or spec['nominal_sha256']!=nominal_hash:
        raise ValueError('Velocity transfer dynamics/nominal pose mismatch')
    saved=torch.load(path,map_location=runner.device,weights_only=False)
    source=saved['actor_state_dict'];dest=runner.alg.actor.state_dict();old_dim,new_dim=445,460
    if source['mlp.0.weight'].shape[1]!=old_dim or dest['mlp.0.weight'].shape[1]!=new_dim:
        raise ValueError('Unexpected velocity observation dimensions')
    for key,value in source.items():
        if dest[key].shape==value.shape:dest[key]=value.clone()
        elif key=='mlp.0.weight':dest[key].zero_();dest[key][:,:old_dim]=value
        elif key in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
            dest[key][...,:old_dim]=value
            # Retain the old normalizer count without giving new features unit
            # variance: these initial scales are explicit m/s engineering priors.
            new_mean=torch.tensor([.06,0.,0.],device=runner.device).repeat(5)
            new_std=torch.tensor([.05,.10,.05],device=runner.device).repeat(5)
            dest[key][...,old_dim:]=new_mean if key.endswith('_mean') else (new_std.square() if key.endswith('_var') else new_std)
        else:raise ValueError(f'Unrecognized expanded parameter: {key}')
    runner.alg.actor.load_state_dict(dest)
    runner.alg.critic.load_state_dict(saved['critic_state_dict'])
    probe=torch.randn(64,new_dim,device=runner.device)
    probe[:,:old_dim]=source['obs_normalizer._mean']+source['obs_normalizer._std']*torch.randn(64,old_dim,device=runner.device)
    old_obs=TensorDict({'actor':probe[:,:old_dim]},batch_size=[64])
    old_model=MLPModel(old_obs,{'actor':['actor']},'actor',22,hidden_dims=(256,128,128),activation='elu',obs_normalization=True,
                       distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'}).to(runner.device)
    old_model.load_state_dict(source);old_model.eval();runner.alg.actor.eval()
    tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        with torch.inference_mode():
            delta=(old_model(old_obs)-runner.alg.actor(TensorDict({'actor':probe},batch_size=[64]))).abs().max().item()
    finally:torch.backends.cuda.matmul.allow_tf32=tf32
    if delta>2e-5:raise AssertionError(f'Initial actions changed during velocity feature expansion: {delta}')
    with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(min=.08,max=.12)
    runner.alg.train_mode()
    return dict(source_checkpoint=str(path.resolve()),source_revision=spec['revision'],actor_max_output_difference=delta,
                old_actor_dim=old_dim,new_actor_dim=new_dim,critic_dim_unchanged=98,extra_feature_weights_initialized_zero=True,
                optimizer_reset=True,iteration_reset=True,simulated_velocity_requires_real_estimator=True,comparison_precision='float32 without TF32')


def load_arm_clearance_transfer(runner,checkpoint,model_hash,nominal_hash,target_revision=24):
    """Explicit R24 transfer across four safe reference-arm targets only."""
    import hashlib
    from .paths import WEEK
    path=Path(checkpoint);spec=json.loads((path.parent/'run.json').read_text(encoding='utf-8'))
    source_path=path.parent/'dynamics/arm_home.json';target_path=WEEK/'model/arm_clearance_home.json'
    if spec['revision'] not in (21,22,23) or spec['model_sha256']!=model_hash:raise ValueError('R24 requires unchanged physical model and full control policy')
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=spec['nominal_sha256'] or hashlib.sha256(target_path.read_bytes()).hexdigest()!=nominal_hash:raise ValueError('Nominal archive hash mismatch')
    old=json.loads(source_path.read_text(encoding='utf-8'));new=json.loads(target_path.read_text(encoding='utf-8'))
    expected={s+'_'+j:v for s in ('l','r') for j,v in (('shoulder_pitch',.15),('elbow_pitch',1.0))}
    changed={k:v for k,v in new['joint_pos'].items() if v!=old['joint_pos'][k]}
    if changed!=expected:raise ValueError('Only the four recorded reference targets may change')
    for key in old:
        if key!='joint_pos' and old[key]!=new[key]:raise ValueError('Unexpected nominal metadata change: '+key)
    result=load_command_warm_start(runner,checkpoint,model_hash,spec['nominal_sha256'],target_revision=target_revision)
    result.update(changed_nominal_pose=True,reference_joint_changes=changed,encoder_and_CAD_zeros_unchanged=True,
                  old_nominal_sha256=spec['nominal_sha256'],new_nominal_sha256=nominal_hash,
                  changed_phase_period=target_revision>=25,physical_mass_inertia_motor_limits_unchanged=True)
    return result


def load_command_warm_start(runner,checkpoint,model_hash,nominal_hash,target_revision=12):
    """Broaden command moments while preserving the straight-walk anchor."""
    path=Path(checkpoint);spec=json.loads((path.parent/'run.json').read_text())
    if spec.get('revision') not in (11,12,13,14,15,16,17,18,19,20,21,22,23) or not spec['whole_body']:
        raise ValueError('Command transfer requires the full-body R11/R12 policy')
    if spec['model_sha256']!=model_hash or spec['nominal_sha256']!=nominal_hash:
        raise ValueError('Command transfer dynamics or nominal pose mismatch')
    saved=torch.load(path,map_location=runner.device,weights_only=False);results={}
    for name,network,dim,outputs in [('actor',runner.alg.actor,460,22),('critic',runner.alg.critic,98,1)]:
        source=saved[name+'_state_dict'];dest={k:v.clone() for k,v in source.items()}
        if source['mlp.0.weight'].shape[1]!=dim:raise ValueError('Unexpected command-transfer schema')
        if name=='actor':
            indices=list(range(30,45))+list(range(435,445))
            anchor=[.06,0,0]*5+[0,1]*5
            widths=[.04,.04,.15]*5+[.15,.05]*5
        else:
            indices=[6,7,8,87,88];anchor=[.06,0,0,0,1];widths=[.04,.04,.15,.15,.05]
        idx=torch.tensor(indices,device=runner.device)
        anchor=torch.tensor(anchor,device=runner.device);widths=torch.tensor(widths,device=runner.device)
        if spec['revision']==11:
            eps=network.obs_normalizer.eps
            old_z=(anchor-source['obs_normalizer._mean'][0,idx])/(source['obs_normalizer._std'][0,idx]+eps)
            dest['mlp.0.bias']+=source['mlp.0.weight'][:,idx]@old_z
            dest['obs_normalizer._mean'][:,idx]=anchor
            dest['obs_normalizer._std'][:,idx]=widths
            dest['obs_normalizer._var'][:,idx]=widths.square()
        network.load_state_dict(dest)
        probe=source['obs_normalizer._mean']+source['obs_normalizer._std']*torch.randn(64,dim,device=runner.device)
        probe[:,idx]=anchor
        old_obs=TensorDict({name:probe},batch_size=[64])
        dist={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'} if name=='actor' else None
        old=MLPModel(old_obs,{name:[name]},name,outputs,hidden_dims=(256,128,128),activation='elu',obs_normalization=True,distribution_cfg=dist).to(runner.device)
        old.load_state_dict(source);old.eval();network.eval()
        tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:
            with torch.inference_mode():delta=(old(old_obs)-network(old_obs)).abs().max().item()
        finally:torch.backends.cuda.matmul.allow_tf32=tf32
        if delta>3e-5:raise AssertionError(f'Walking anchor changed: {name} {delta}')
        results[name+'_anchor_max_difference']=delta
    with torch.no_grad():runner.alg.actor.distribution.std_param.clamp_(min=.04 if target_revision>=21 else (.06 if target_revision>=16 else (.04 if target_revision>=13 else .06)),max=.07 if target_revision>=21 else (.09 if target_revision>=16 else (.06 if target_revision>=13 else .10)))
    runner.alg.train_mode()
    results.update(source=str(path.resolve()),source_revision=spec['revision'],optimizer_reset=True,
                   changed_nominal_pose=False,changed_phase_period=target_revision>=25,command_normalization_rebased=spec['revision']==11,
                   phase_input_gated_by_motion=target_revision>=13,
                   comparison_scope='64 varied proprioception probes at vx=.06,vy=wz=heading_error=0; TF32 disabled')
    return results
