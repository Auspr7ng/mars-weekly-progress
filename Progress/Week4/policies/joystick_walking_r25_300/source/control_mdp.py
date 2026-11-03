"""Command transitions and foot-space shaping for speed/stop/yaw training.

These are reward targets only. All joint targets still come from PPO.
"""
import torch


def slew_rates(revision):
    return (.5,.5,.8) if revision>=22 else (.25,.25,.35)


def sample_commands(term,env_ids):
    if term._env.cfg.sunday_revision>=23:
        count=len(env_ids);draw=torch.rand(count,device=term.device);term.target[env_ids]=0
        edge=torch.rand(count,device=term.device)<.25
        forward=torch.empty(count,device=term.device).uniform_(.03,.15)
        backward=torch.empty(count,device=term.device).uniform_(.02,.06)
        reverse=(draw>=.55)&(draw<.70)
        speed=torch.where(reverse,-torch.where(edge,.06,backward),torch.where(edge,.15,forward))
        moving=(draw>=.20)&(draw<.85)
        yaw=torch.empty(count,device=term.device).uniform_(.08,.4)
        yaw=torch.where(torch.rand(count,device=term.device)<.25,.4,yaw)
        yaw*=torch.where(torch.rand(count,device=term.device)<.5,-1.,1.)
        term.target[env_ids,0]=torch.where(moving,speed,0.)
        term.target[env_ids,2]=torch.where(draw>=.70,yaw,0.)
        anchor=(draw>=.20)&(draw<.30);term.target[env_ids[anchor],0]=.06
        return
    if term._env.cfg.sunday_revision>=14:
        count=len(env_ids);draw=torch.rand(count,device=term.device)
        term.target[env_ids]=0
        # Balance both turn directions instead of the previous3% pure-yaw total.
        progress=min(1.,term._env.common_step_counter/6400.) if term._env.cfg.sunday_revision>=22 else 0.
        speed=torch.empty(count,device=term.device).uniform_(.03,.08+.07*progress)
        if term._env.cfg.sunday_revision>=22:
            reverse=(draw>=.40)&(draw<.50)
            backwards=torch.empty(count,device=term.device).uniform_(.02,.02001+.04*progress)
            speed=torch.where(reverse,-backwards,speed)
        moving=draw>=.25;turning=draw>=.55;pure=draw>=.80
        yaw=torch.empty(count,device=term.device).uniform_(.08,.12+.28*progress)
        yaw*=torch.where(torch.rand(count,device=term.device)<.5,-1.,1.)
        term.target[env_ids,0]=torch.where(moving&~pure,speed,0.)
        term.target[env_ids,2]=torch.where(turning,yaw,0.)
        anchor=(draw>=.25)&(draw<.40)
        term.target[env_ids[anchor],0]=.06
        return
    progress=max(0.,min(1.,(term._env.common_step_counter-3200)/6400))
    count=len(env_ids)
    draw=torch.rand(count,device=term.device)
    term.target[env_ids]=0
    speed=torch.empty(count,device=term.device).uniform_(.045-.015*progress,.075+.010*progress)
    stopping_focus=term._env.cfg.sunday_revision>=13
    moving=draw>=(.50 if stopping_focus else .28)
    term.target[env_ids,0]=torch.where(moving,speed,0.)
    turning=draw>=(.90 if stopping_focus else .65)
    yaw_cap=(.08+.10*progress) if stopping_focus else (.04+.14*progress)
    yaw=torch.empty(count,device=term.device).uniform_(.5*yaw_cap,yaw_cap)
    yaw*=torch.where(torch.rand(count,device=term.device)<.5,-1.,1.)
    term.target[env_ids,2]=torch.where(turning,yaw,0.)
    # Last 8% are pure yaw: stepping in place with no translation command.
    term.target[env_ids[draw>=(.97 if stopping_focus else .92)],0]=0
    # Retain an easy initial walking anchor in most first episodes.
    first=term.command_counter[env_ids]==0
    anchor=first&(torch.rand(count,device=term.device)<(.50 if stopping_focus else .65))
    if stopping_focus:term.target[env_ids[first]]=0
    term.target[env_ids[anchor]]=torch.tensor([.06,0,0],device=term.device)


def command_activity(env):
    c=env.command_manager.get_command('twist')
    # Gate normalization: a pure 0.08rad/s turn warrants a full stepping gait.
    return c[:,:2].norm(dim=1)+.375*c[:,2].abs()


def gait_weight(env):
    return env.command_manager.get_term('twist').motion_gate


def foot_targets(env):
    """Body-frame foot placement from a small body twist; no root actuation."""
    time=env.episode_length_buf*env.step_dt
    p=(time/1.4+.075)%1
    period=1.4
    if env.cfg.sunday_revision>=25:
        term=env.command_manager.get_term('twist');p=term.gait_phase;period=term.gait_period[:,None]
    ph=torch.stack([p,(p+.5)%1],dim=1)
    stance=ph<.65;u=((ph-.65)/.35).clamp(0,1)
    ramp=(time/.7).clamp(0,1)*gait_weight(env)
    nominal=torch.tensor(env.cfg.sunday_sole_xy,device=env.device)
    c=env.command_manager.get_command('twist')
    # Opposite foot sweep induced by yaw: v + omega x r.
    vx=c[:,0,None]-c[:,2,None]*nominal[None,:,1]
    vy=c[:,1,None]+c[:,2,None]*nominal[None,:,0]
    def path(velocity):
        span=velocity*period*.65;back=-velocity*period*.35
        swing=(2*u**3-3*u**2+1)*(-span/2)+(u**3-2*u**2+u)*back+(-2*u**3+3*u**2)*(span/2)+(u**3-u**2)*back
        return torch.where(stance,span/2-velocity*period*ph,swing)*ramp[:,None]
    dx=path(vx);dy=path(vy)
    clearance=.024 if env.cfg.sunday_revision>=14 else .020
    z=clearance*torch.sin(torch.pi*u).square()*(~stance)*ramp[:,None]
    amplitude=env.command_manager.get_term('twist').sway_amplitude if env.cfg.sunday_revision>=15 else .018
    sway=amplitude*torch.cos(2*torch.pi*(p-.325))*ramp
    relative_yaw=path(c[:,2,None].expand_as(vx))
    return stance,u,nominal[None,:,0]+dx,nominal[None,:,1]+dy-sway[:,None],z,sway,relative_yaw


def sway_velocity(env):
    time=env.episode_length_buf*env.step_dt
    angle=2*torch.pi*(time/1.4+.075-.325)
    period=1.4
    if env.cfg.sunday_revision>=25:
        term=env.command_manager.get_term('twist');angle=2*torch.pi*(term.gait_phase-.325);period=term.gait_period
    ramp=(time/.7).clamp(0,1);ramp_dot=(time<.7).float()/.7
    term=env.command_manager.get_term('twist');gate=term.motion_gate
    amplitude=term.sway_amplitude if env.cfg.sunday_revision>=15 else .018
    amplitude_dot=term.sway_amplitude_dot if env.cfg.sunday_revision>=15 else 0.
    return amplitude*(-2*torch.pi/period*torch.sin(angle)*ramp*gate+torch.cos(angle)*(ramp_dot*gate+ramp*term.motion_gate_dot))+amplitude_dot*torch.cos(angle)*ramp*gate


def reward(env,kind):
    from .mdp import contact,sole_geometry
    from mjlab.utils.lab_api.math import matrix_from_quat
    d=env.scene['robot'].data
    idle=(command_activity(env)<.003)&(gait_weight(env)<.15)
    if kind=='quality_in_place_cost':
        c=env.command_manager.get_command('twist')
        pure=(c[:,:2].norm(dim=1)<.005)&(c[:,2].abs()>.05)
        cost=(d.root_link_lin_vel_b[:,0]/.025).square()
        cost+=((d.root_link_lin_vel_b[:,1]-sway_velocity(env))/.04).square()
        return cost.clamp(max=10)*pure*gait_weight(env)
    if kind=='quality_stop_transition':
        term=env.command_manager.get_term('twist')
        requested=term.target if term.override is None else torch.as_tensor(term.override,device=env.device)
        stop_requested=requested.abs().sum(dim=-1)<1e-6
        _,lowest=sole_geometry(env)
        ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
        cost=(d.root_link_lin_vel_b[:,:2]/.04).square().sum(dim=1)+(d.root_link_ang_vel_b[:,2]/.12).square()
        cost+=.2*((lowest.clamp(min=0)/.02).square()+(d.site_lin_vel_w[:,ids].norm(dim=-1)/.06).square()).mean(dim=1)
        return cost.clamp(max=10)*stop_requested*(1-gait_weight(env))
    if kind=='quality_stop':
        quiet=(d.root_link_lin_vel_b[:,:2]/.025).square().sum(dim=1)+(d.root_link_ang_vel_b[:,2]/.10).square()
        return torch.exp(-quiet)*contact(env).all(dim=1)*idle
    if kind=='quality_stop_cost':
        action=env.action_manager.get_term('residual')
        cost=(d.root_link_lin_vel_b[:,:2]/.03).square().sum(dim=1)+(d.root_link_ang_vel_b[:,2]/.10).square()
        cost+=.2*(d.joint_vel[:,action.ids]/.5).square().mean(dim=1)
        return cost.clamp(max=10)*idle
    if kind=='quality_stop_feet':
        _,lowest=sole_geometry(env)
        ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
        velocity=d.site_lin_vel_w[:,ids]
        cost=(lowest.clamp(min=0)/.01).square().clamp(max=4)
        cost+=(velocity.norm(dim=-1)/.04).square().clamp(max=8)
        return cost.mean(dim=1)*idle
    if kind=='quality_stop_pose':
        action=env.action_manager.get_term('residual')
        error=(d.joint_pos[:,action.ids]-d.default_joint_pos[:,action.ids])/.18
        return torch.exp(-error.square().mean(dim=1))*idle
    if kind=='quality_foot_heading':
        *_,yaw_target=foot_targets(env)
        ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
        r=matrix_from_quat(d.site_quat_w[:,ids]);root=matrix_from_quat(d.root_link_quat_w)
        # Original foot toe direction is local -Y; CAD is oriented to forward +X.
        actual=torch.atan2(-r[...,1,1],-r[...,0,1])-torch.atan2(root[:,1,0],root[:,0,0])[:,None]
        error=torch.atan2(torch.sin(actual-yaw_target),torch.cos(actual-yaw_target))
        return torch.exp(-(error/.18).square()).mean(dim=1)*gait_weight(env)
    raise ValueError(kind)
