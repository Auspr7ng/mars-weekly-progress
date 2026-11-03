"""Revision-5 gait shaping: foot paths, weight transfer, planting and quiet torso."""
import math
import torch
from mjlab.utils.lab_api.math import matrix_from_quat


def clearance_shortfall_cost(lowest,active,revision):
    scale=.008 if revision>=16 else (.004 if revision>=15 else .008)
    shortfall=((.015*active[:,None]-lowest)/scale).clamp(min=0)
    return shortfall.square().clamp(max=25. if revision>=16 else 9.)


def load_transfer_cost(force_fraction,stance):
    desired=stance.float()/stance.sum(dim=-1,keepdim=True).clamp(min=1)
    return ((force_fraction-desired)/.4).square().mean(dim=-1).clamp(max=10)


def plan(env):
    if env.cfg.sunday_revision>=12:
        from .control_mdp import foot_targets
        stance,u,x,_,z,y,_=foot_targets(env)
        nominal=torch.tensor(env.cfg.sunday_sole_xy,device=env.device)
        return stance,u,x-nominal[None,:,0],z,y
    time=env.episode_length_buf*env.step_dt
    p=(time/1.4+.075)%1
    ph=torch.stack([p,(p+.5)%1],dim=1)
    stance=ph<.65
    u=((ph-.65)/.35).clamp(0,1)
    v=env.command_manager.get_command('twist')[:,0].clamp(0,.15)
    ramp=(time/.7).clamp(0,1)
    span=(v*1.4*.65)[:,None]
    backward=-(v*1.4*.35)[:,None]
    swing=(2*u**3-3*u**2+1)*(-span/2)+(u**3-2*u**2+u)*backward+(-2*u**3+3*u**2)*(span/2)+(u**3-u**2)*backward
    x=torch.where(stance,span/2-v[:,None]*1.4*ph,swing)*ramp[:,None]
    z=.020*torch.sin(torch.pi*u).square()*(~stance)*ramp[:,None]
    y=(.018 if env.cfg.sunday_revision>=9 else .028)*torch.cos(2*torch.pi*(p-.325))*ramp
    return stance,u,x,z,y


def lateral_velocity_target(env):
    """Time derivative of the planned lateral transfer, in the heading frame."""
    if env.cfg.sunday_revision>=12:
        from .control_mdp import sway_velocity
        return sway_velocity(env)
    time=env.episode_length_buf*env.step_dt
    angle=2*torch.pi*(time/1.4+.075-.325)
    ramp=(time/.7).clamp(0,1)
    ramp_dot=(time<.7).float()/.7
    amplitude=.018 if env.cfg.sunday_revision>=9 else .028
    velocity=amplitude*(-2*torch.pi/1.4*torch.sin(angle)*ramp+torch.cos(angle)*ramp_dot)
    active=env.command_manager.get_command('twist')[:,:2].norm(dim=1)>.005
    return velocity*active


def reward(env,kind):
    if kind=='quality_foot_separation':
        from .clearance_mdp import foot_clearance_cost
        return foot_clearance_cost(env)
    if kind=='quality_hand_clearance':
        from .clearance_mdp import hand_clearance_cost
        return hand_clearance_cost(env)
    if kind=='quality_torso_oscillation':
        from .clearance_mdp import torso_oscillation_cost
        return torso_oscillation_cost(env)
    if kind in ('quality_stop','quality_stop_cost','quality_stop_feet','quality_stop_pose','quality_stop_transition','quality_in_place_cost','quality_foot_heading'):
        from .control_mdp import reward as control_reward
        return control_reward(env,kind)
    from .mdp import sole_geometry,foot_forces
    data=env.scene['robot'].data
    stance,u,x,z,y=plan(env)
    pos,lowest=sole_geometry(env)
    active=env.command_manager.get_command('twist')[:,:2].norm(dim=1)>(.005 if env.cfg.sunday_revision>=7 else .03)
    if env.cfg.sunday_revision>=12:
        from .control_mdp import gait_weight
        active=gait_weight(env)
    force=foot_forces(env).norm(dim=-1)
    support=force>.10*env.cfg.sunday_mass*9.81
    if kind=='quality_load_transfer':return load_transfer_cost(force/(env.cfg.sunday_mass*9.81),stance)*active
    ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
    rotation=matrix_from_quat(data.site_quat_w[:,ids])
    tilt=torch.acos(rotation[...,2,2].clamp(-1,1))
    if kind=='quality_clearance':
        if env.cfg.sunday_revision>=6:return (1-((lowest-z).abs()/.020)).clamp(-1,1).mean(dim=1)*active
        return torch.exp(-((lowest-z)/.006).square()).mean(dim=1)*active
    if kind=='quality_clearance_margin':
        # Dense penalty for shuffling through the middle of a scheduled swing.
        # This changes only the reward; no contact or action is overridden.
        middle=(~stance)&(u>.30)&(u<.70)
        # R15 capped the shortfall before squaring: near-ground samples all
        # received the same maximum cost. Keep sensitivity throughout lift-off.
        return (clearance_shortfall_cost(lowest,active,env.cfg.sunday_revision)*middle).mean(dim=1)*active
    if kind=='quality_foot_path':
        base_rotation=matrix_from_quat(data.root_link_quat_w)
        yaw=torch.atan2(base_rotation[:,1,0],base_rotation[:,0,0])
        offset=pos-data.root_link_pos_w[:,None]
        rx=offset[...,0]*torch.cos(yaw[:,None])+offset[...,1]*torch.sin(yaw[:,None])
        ry=-offset[...,0]*torch.sin(yaw[:,None])+offset[...,1]*torch.cos(yaw[:,None])
        nominal=torch.tensor(env.cfg.sunday_sole_xy,device=env.device)
        if env.cfg.sunday_revision>=12:
            from .control_mdp import foot_targets
            _,_,tx,ty,*_=foot_targets(env)
            error=(rx-tx).abs()/.04+(ry-ty).abs()/.03
            return (1-.5*error).clamp(-1,1).mean(dim=1)*active
        if env.cfg.sunday_revision>=6:
            error=(rx-nominal[None,:,0]-x).abs()/.04+(ry-nominal[None,:,1]+y[:,None]).abs()/.03
            return (1-.5*error).clamp(-1,1).mean(dim=1)*active
        error=((rx-nominal[None,:,0]-x)/.020).square()+((ry-nominal[None,:,1]+y[:,None])/.018).square()
        return torch.exp(-error).mean(dim=1)*active
    if kind in ('quality_sway','quality_transfer_cost'):
        actual=data.root_link_pos_w[:,1]-env.scene.env_origins[:,1]
        if env.cfg.sunday_revision>=6:actual=data.root_link_pos_w[:,1]-pos[...,1].mean(dim=1)
        if env.cfg.sunday_revision>=12:
            root=matrix_from_quat(data.root_link_quat_w)
            yaw=torch.atan2(root[:,1,0],root[:,0,0])
            delta=data.root_link_pos_w[:,:2]-pos[...,:2].mean(dim=1)
            actual=-torch.sin(yaw)*delta[:,0]+torch.cos(yaw)*delta[:,1]
        if kind=='quality_transfer_cost':return ((actual-y)/.012).square().clamp(max=6)*active
        return torch.exp(-((actual-y)/.020).square())*active
    if kind=='quality_lane':
        actual=data.root_link_pos_w[:,1]-env.scene.env_origins[:,1]
        return ((actual-y)/.05).square().clamp(max=10)*active
    if kind=='quality_plant_slip':
        speed=data.site_lin_vel_w[:,ids,:2].norm(dim=-1)
        mask=support if env.cfg.sunday_revision>=8 else support*stance
        return ((speed/.03).square().clamp(max=10)*mask).mean(dim=1)*active
    if kind=='quality_landing_brake':
        velocity=data.site_lin_vel_w[:,ids]
        late=(~stance)&(u>.8)&(lowest<.007)
        cost=(velocity[...,:2]/.05).square().sum(dim=-1)+(velocity[...,2]/.1).square()
        return (cost.clamp(max=10)*late).mean(dim=1)*active
    if kind=='quality_foot_flat':return ((tilt/.12).square().clamp(max=10)*support).mean(dim=1)
    if kind=='quality_swing_contact':return (support*(~stance)*(u>.15)*(u<.85)).float().mean(dim=1)*active
    if kind=='quality_impact':
        peak=env.scene['feet_contact'].data.force_history.norm(dim=-1).amax(dim=-1)/(env.cfg.sunday_mass*9.81)
        return (peak-1.3).clamp(min=0,max=5).square().mean(dim=1)
    if kind=='quality_joint_acc':
        action=env.action_manager.get_term('residual')
        denominator=40*(env.step_dt if env.cfg.sunday_revision>=8 else 1.)
        return (((data.joint_vel[:,action.ids]-action.previous_joint_velocity)/denominator).square()).mean(dim=1)
    if kind=='quality_phase':
        if env.cfg.sunday_revision>=16:
            clearance=(lowest.clamp(min=0)/(.015*active[:,None]).clamp(min=.005)).clamp(max=1)
            score=torch.where(stance,support.float(),clearance)
            return score.prod(dim=1)*active
        air=(~support)&(lowest>.005)
        score=torch.where(stance,support,air).float()
        return score.prod(dim=1)*active
    if kind=='quality_arms':
        vx=env.command_manager.get_command('twist')[:,0]
        span=((vx.abs() if env.cfg.sunday_revision>=22 else vx).clamp(min=.005)*1.4*.65)
        if env.cfg.sunday_revision>=25:span=vx.abs().clamp(min=.005)*env.command_manager.get_term('twist').gait_period*.65
        swing=.15*((x[:,0]-x[:,1])/span).clamp(-1,1)
        errors=[]
        for side in ('l','r'):
            shoulder=env.scene['robot'].joint_names.index(side+'_shoulder_roll')
            elbow=env.scene['robot'].joint_names.index(side+'_elbow_pitch')
            errors.append(((data.joint_pos[:,shoulder]-data.default_joint_pos[:,shoulder]-swing)/.15).square())
            errors.append(((data.joint_pos[:,elbow]-data.default_joint_pos[:,elbow])/.12).square())
        return torch.exp(-torch.stack(errors,dim=1).mean(dim=1))*active
    raise ValueError(kind)


def motion_factor(env):
    """Do not pay full locomotion rewards for moving while planted feet slide."""
    from .mdp import sole_geometry,foot_forces
    stance,u,x,z,y=plan(env)
    _,lowest=sole_geometry(env)
    data=env.scene['robot'].data
    ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
    speed=data.site_lin_vel_w[:,ids,:2].norm(dim=-1)
    support=foot_forces(env).norm(dim=-1)>.1*env.cfg.sunday_mass*9.81
    plant=torch.exp(-(speed/.04).square())*support
    plant=torch.where(stance,plant,1.).amin(dim=1)
    swing_ratio=(lowest.clamp(min=0)/z.clamp(min=.006)).clamp(0,1)
    swing_ratio=torch.where((~stance)&(u>.12)&(u<.88),swing_ratio,1.).amin(dim=1)
    factor=.15+.85*plant*swing_ratio
    if env.cfg.sunday_revision>=11:
        active=env.command_manager.get_command('twist')[:,:2].norm(dim=1)>.005
        if env.cfg.sunday_revision>=12:
            from .control_mdp import gait_weight
            active=gait_weight(env)>.08
        factor=torch.where(active,factor,torch.ones_like(factor))
    return factor
