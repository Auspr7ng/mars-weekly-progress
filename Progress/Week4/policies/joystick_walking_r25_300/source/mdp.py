"""Actions, command sampling, rewards and failure checks for Sunday A1."""
from dataclasses import dataclass
import json
import math

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from .paths import HOME


@dataclass(kw_only=True)
class ResidualActionCfg(ActionTermCfg):
    joint_names: tuple[str, ...]

    def build(self, env):
        return ResidualAction(self, env)


class ResidualAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        h=json.loads(HOME.read_text())
        self.names=tuple(cfg.joint_names)
        self.ids=torch.tensor([self._entity.joint_names.index(n) for n in self.names],device=self.device)
        self.scales=torch.tensor([h['actuators'][n]['scale'] for n in self.names],device=self.device)
        if env.cfg.sunday_revision>=2:
            for i,n in enumerate(self.names):
                if n in h['leg_order']:
                    self.scales[i]=.65 if 'knee' in n else (.45 if 'roll' in n else .35)
                elif env.cfg.sunday_revision>=7 and n.endswith('shoulder_roll'):self.scales[i]=.25
                elif env.cfg.sunday_revision>=7 and n.endswith('elbow_pitch'):self.scales[i]=.15
        self.speeds=torch.tensor([h['actuators'][n]['speed'] for n in self.names],device=self.device)
        if env.cfg.sunday_revision>=23:
            from .action_ranges import expanded_joints
            self.expanded_mask=torch.tensor(expanded_joints(self.names,env.cfg.sunday_revision),device=self.device)
        self.torque_limits=torch.tensor([h['actuators'][n]['effort'] for n in self._entity.actuator_names],device=self.device)
        self._raw=torch.zeros(self.num_envs,len(self.names),device=self.device)
        self.targets=self._entity.data.default_joint_pos.clone()
        self.previous_joint_velocity=self._entity.data.joint_vel[:,self.ids].clone()

    @property
    def action_dim(self):
        return len(self.names)

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Nonfinite policy action; stopping before physics/PPO contamination')
        self._raw[:]=actions.clamp(-1,1)
        self.previous_joint_velocity[:]=self._entity.data.joint_vel[:,self.ids]
        default=self._entity.data.default_joint_pos
        scales=self.scales
        if 23<=self._env.cfg.sunday_revision<=24:
            c=self._env.command_manager.get_command('twist')
            level=torch.maximum(((c[:,0].abs()-.08)/.07).clamp(0,1),((c[:,2].abs()-.12)/.28).clamp(0,1))
            if self._env.cfg.sunday_revision>=24:level=((c[:,0].abs()-.08)/.07).clamp(0,1)
            if self._env.cfg.sunday_action_gain_curriculum:level=level*min(1.,self._env.common_step_counter/3200.)
            scales=scales*(1.+.5*level[:,None]*self.expanded_mask)
        requested=default[:,self.ids]+self._raw*scales
        step=self.speeds*self._env.step_dt
        previous=self.targets[:,self.ids]
        if self._env.cfg.sunday_revision>=3:
            requested=previous+.3*(requested-previous)
        requested=torch.maximum(torch.minimum(requested,previous+step),previous-step)
        limits=self._entity.data.soft_joint_pos_limits[:,self.ids]
        self.targets[:]=default
        self.targets[:,self.ids]=torch.maximum(torch.minimum(requested,limits[...,1]),limits[...,0])

    def apply_actions(self):
        self._entity.set_joint_position_target(self.targets)

    def reset(self, env_ids=None):
        if env_ids is None:env_ids=slice(None)
        self._raw[env_ids]=0
        self.targets[env_ids]=self._entity.data.default_joint_pos[env_ids]


@dataclass(kw_only=True)
class TwistCfg(CommandTermCfg):
    stage: str='stand'

    def build(self, env):
        return Twist(self,env)


class Twist(CommandTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.value=torch.zeros(self.num_envs,3,device=self.device)
        self.target=torch.zeros_like(self.value)
        self.target_yaw=torch.zeros(self.num_envs,device=self.device)
        self.override=None
        self.motion_gate=torch.zeros(self.num_envs,device=self.device)
        self.motion_gate_dot=torch.zeros_like(self.motion_gate)
        self.sway_amplitude=torch.full_like(self.motion_gate,.018)
        self.sway_amplitude_dot=torch.zeros_like(self.motion_gate)
        self.gait_phase=torch.full_like(self.motion_gate,.075)
        self.gait_period=torch.full_like(self.motion_gate,1.4)
        self.metrics['velocity_error']=torch.zeros(self.num_envs,device=self.device)

    @property
    def command(self):return self.value

    def compute(self,dt):
        self.command_dt=dt
        super().compute(dt)

    def reset(self,env_ids):
        result=super().reset(env_ids)
        if self._env.cfg.sunday_revision>=4:
            # Reset events use yaw=0; derived link poses are not forwarded yet.
            self.target_yaw[env_ids]=0
        if self._env.cfg.sunday_revision>=12:
            self.value[env_ids]=0
            self.motion_gate[env_ids]=0;self.motion_gate_dot[env_ids]=0
            self.sway_amplitude[env_ids]=.018;self.sway_amplitude_dot[env_ids]=0
            self.gait_phase[env_ids]=.075;self.gait_period[env_ids]=1.4
        return result

    def _resample_command(self,env_ids):
        if self._env.cfg.sunday_revision>=12 and self.cfg.stage=='control':
            from .control_mdp import sample_commands
            sample_commands(self,env_ids)
            return
        ranges={'stand':[(0,0),(0,0),(0,0)],'forward':[(.05,.15),(0,0),(0,0)],
                'walk':[(-.1,.25),(-.05,.05),(-.3,.3)],'omni':[(-.15,.35),(-.15,.15),(-.6,.6)]}[self.cfg.stage]
        for k,(lo,hi) in enumerate(ranges):
            self.target[env_ids,k]=torch.empty(len(env_ids),device=self.device).uniform_(lo,hi)
        if self._env.cfg.sunday_revision>=2 and self.cfg.stage=='forward':
            self.target[env_ids,0]=.08
        standing=torch.rand(len(env_ids),device=self.device)<(0. if self._env.cfg.sunday_revision>=2 else .2)
        self.target[env_ids[standing]]=0
        if self._env.cfg.sunday_revision==1:self.value[env_ids]=0

    def _update_command(self):
        # reset() calls compute(dt=0): a new episode must not integrate a turn.
        if self._env.cfg.sunday_revision>=12 and self.command_dt==0:return
        if self.override is not None:
            target=torch.as_tensor(self.override,device=self.device,dtype=torch.float32).expand_as(self.target)
        else:target=self.target
        if self._env.cfg.sunday_revision>=8 and self.cfg.stage=='forward' and self.override is None:
            self.target[:,0]=.06
        elif self._env.cfg.sunday_revision>=7 and self.cfg.stage=='forward' and self.override is None:
            # 100 PPO iterations of slow stepping, then 200 of gradual acceleration.
            fraction=max(0.,min(1.,(self._env.common_step_counter-3200)/6400))
            self.target[:,0]=.02+.08*fraction
        from .control_mdp import slew_rates
        rates=torch.tensor(slew_rates(self._env.cfg.sunday_revision) if self._env.cfg.sunday_revision>=12 else [.5,.5,1.0],device=self.device)*self._env.step_dt
        self.value+=torch.maximum(torch.minimum(target-self.value,rates),-rates)
        if self._env.cfg.sunday_revision>=12:
            activity=self.value[:,:2].norm(dim=1)+.375*self.value[:,2].abs()
            desired=(activity/.03).clamp(0,1)
            updated=self.motion_gate+(1-math.exp(-self._env.step_dt/.18))*(desired-self.motion_gate)
            self.motion_gate_dot=(updated-self.motion_gate)/self._env.step_dt
            self.motion_gate[:]=updated
            if self._env.cfg.sunday_revision>=15:
                yaw_activity=.375*self.value[:,2].abs()
                ratio=yaw_activity/(self.value[:,:2].norm(dim=1)+yaw_activity+1e-6)
                desired_amplitude=.018+.010*ratio
                amplitude=self.sway_amplitude+(1-math.exp(-self._env.step_dt/.18))*(desired_amplitude-self.sway_amplitude)
                self.sway_amplitude_dot=(amplitude-self.sway_amplitude)/self._env.step_dt
                self.sway_amplitude[:]=amplitude
        if self._env.cfg.sunday_revision>=25:
            from .cadence import torch_period
            desired=torch_period(self.value)
            if self._env.cfg.sunday_cadence_curriculum:desired=1.4+(desired-1.4)*min(1.,self._env.common_step_counter/3200.)
            self.gait_period[:]=desired
            self.gait_phase[:]=(self.gait_phase+self._env.step_dt/desired)%1.
        if self._env.cfg.sunday_revision>=4:
            self.target_yaw+=self.value[:,2]*self._env.step_dt

    def _update_metrics(self):
        err=torch.linalg.vector_norm(self._env.scene['robot'].data.root_link_lin_vel_b[:,:2]-self.value[:,:2],dim=1)
        self.metrics['velocity_error']+=err*self._env.step_dt/self.cfg.resampling_time_range[1]


def command(env):return env.command_manager.get_command('twist')


def heading(env):
    r=matrix_from_quat(env.scene['robot'].data.root_link_quat_w)
    error=env.command_manager.get_term('twist').target_yaw-torch.atan2(r[:,1,0],r[:,0,0])
    return torch.stack([torch.sin(error),torch.cos(error)],dim=1)


def feet(env):
    return env.scene['feet_contact'].data


def foot_forces(env):
    # Sensor primaries follow model order (right,left); sites use (left,right).
    sensor=env.scene['feet_contact']
    names=[n.split('/')[-1] for n in sensor.primary_names]
    order=[names.index('l_foot_link_1'),names.index('r_foot_link_1')]
    if env.cfg.sunday_revision>=2:
        return sensor.data.force_history[:,order].mean(dim=2)
    return sensor.data.force[:,order]


def contact(env):
    # netforce may point toward or away from primary; magnitudes are sign-invariant.
    f=foot_forces(env)
    return torch.linalg.vector_norm(f,dim=-1)>.03*env.cfg.sunday_mass*9.81


def height(env):
    return env.scene['robot'].data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]


def sole_geometry(env):
    data=env.scene['robot'].data
    ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
    pos=data.site_pos_w[:,ids]
    rotation=matrix_from_quat(data.site_quat_w[:,ids])
    lowest=pos[...,2]-env.scene.env_origins[:,None,2]-rotation[...,2,0].abs()*.0525-rotation[...,2,1].abs()*.09
    return pos,lowest


def phase(env):
    period=1.4 if env.cfg.sunday_revision>=5 else (1. if env.cfg.sunday_revision>=2 else .75)
    angle=(env.episode_length_buf*env.step_dt/period+(.075 if env.cfg.sunday_revision>=5 else 0))*(2*torch.pi)
    if env.cfg.sunday_revision>=25:angle=env.command_manager.get_term('twist').gait_phase*(2*torch.pi)
    signal=torch.stack([torch.sin(angle),torch.cos(angle)],dim=1)
    if env.cfg.sunday_revision>=13:
        # Phase is a walking cue, not a demand to keep marching at zero command.
        signal*=env.command_manager.get_term('twist').motion_gate[:,None]
    return signal


def foot_state(env):
    data=env.scene['robot'].data
    ids=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
    return torch.cat([data.site_pos_w[:,ids,2]-env.scene.env_origins[:,None,2],contact(env).float(),
                      torch.linalg.vector_norm(foot_forces(env),dim=-1)/(env.cfg.sunday_mass*9.81)],dim=1)


def reward(env,kind):
    if kind.startswith('quality_'):
        from .quality_mdp import reward as quality_reward
        return quality_reward(env,kind)
    d=env.scene['robot'].data
    cmd=command(env)
    speed=torch.linalg.vector_norm(cmd[:,:2],dim=1)
    active=(speed>(.005 if env.cfg.sunday_revision>=7 else .03))|(cmd[:,2].abs()>.08)
    if env.cfg.sunday_revision>=12:
        from .control_mdp import command_activity,gait_weight
        active=(command_activity(env)>.003)|(gait_weight(env)>.08)
    act=env.action_manager.get_term('residual')
    ids=act.ids
    if kind=='track_xy':
        sigma=.05 if env.cfg.sunday_revision>=2 else .15
        if env.cfg.sunday_revision>=5:
            squared_error=((d.root_link_lin_vel_b[:,0]-cmd[:,0])/sigma).square()
            if env.cfg.sunday_revision>=11:
                from .quality_mdp import lateral_velocity_target
                squared_error+=((d.root_link_lin_vel_b[:,1]-cmd[:,1]-lateral_velocity_target(env))/sigma).square()
            value=torch.exp(-squared_error)
            if env.cfg.sunday_revision>=7:
                from .quality_mdp import motion_factor
                value*=motion_factor(env)
            return value
        return torch.exp(-torch.sum((d.root_link_lin_vel_b[:,:2]-cmd[:,:2])**2,dim=1)/sigma**2)
    if kind=='progress':
        denominator=speed.clamp(min=.005 if env.cfg.sunday_revision>=8 else .04)
        projected=(d.root_link_lin_vel_b[:,:2]*cmd[:,:2]).sum(dim=1)/denominator
        value=(projected/denominator).clamp(-1,1)*active
        if env.cfg.sunday_revision>=7:
            from .quality_mdp import motion_factor
            value*=motion_factor(env)
        return value
    if kind=='under_speed':
        denominator=speed.clamp(min=.005 if env.cfg.sunday_revision>=8 else .04)
        projected=(d.root_link_lin_vel_b[:,:2]*cmd[:,:2]).sum(dim=1)/denominator
        return ((.7*speed-projected)/denominator).clamp(0,2)*active*(env.episode_length_buf>50)
    if kind=='failure':return env.termination_manager.terminated.float()/env.step_dt
    if kind=='velocity_cost':
        if env.cfg.sunday_revision>=5:
            longitudinal_scale=.04 if env.cfg.sunday_revision>=25 else .08
            cost=((d.root_link_lin_vel_b[:,0]-cmd[:,0])/longitudinal_scale).square()
            if env.cfg.sunday_revision>=25:cost*=torch.where(cmd[:,0]<-.005,2.,1.)
            if env.cfg.sunday_revision>=11:
                from .quality_mdp import lateral_velocity_target
                lateral_scale=.04 if env.cfg.sunday_revision>=14 else .08
                cost+=((d.root_link_lin_vel_b[:,1]-cmd[:,1]-lateral_velocity_target(env))/lateral_scale).square()
            return cost.clamp(max=10)*active
        return (torch.sum((d.root_link_lin_vel_b[:,:2]-cmd[:,:2])**2,dim=1)/.08**2).clamp(max=10)*active
    if kind=='yaw_cost':return ((d.root_link_ang_vel_b[:,2]-cmd[:,2])/(.10 if env.cfg.sunday_revision>=25 else (.15 if env.cfg.sunday_revision>=12 else .3))).square().clamp(max=10)
    if kind=='heading_cost':
        h=heading(env)
        return (torch.atan2(h[:,0],h[:,1])/.2).square().clamp(max=10)
    if kind=='lateral_speed':return (d.root_link_lin_vel_b[:,1]/.03).square().clamp(max=10)
    if kind=='track_yaw':
        value=torch.exp(-(d.root_link_ang_vel_b[:,2]-cmd[:,2])**2/(.12 if env.cfg.sunday_revision>=12 else .30)**2)
        if env.cfg.sunday_revision>=14:
            from .quality_mdp import motion_factor
            value*=motion_factor(env)
        return value
    if kind=='upright':
        cos=(-d.projected_gravity_b[:,2]).clamp(-1,1)
        if env.cfg.sunday_revision>=18:
            term=env.command_manager.get_term('twist')
            ratio=((term.sway_amplitude-.018)/.010).clamp(0,1)
            time=env.episode_length_buf*env.step_dt
            lean=-.04*ratio*torch.cos(2*torch.pi*(time/1.4+.075-.325))*term.motion_gate
            if env.cfg.sunday_revision>=25:lean=-.04*ratio*torch.cos(2*torch.pi*(term.gait_phase-.325))*term.motion_gate
            cos=(-torch.sin(lean)*d.projected_gravity_b[:,1]-torch.cos(lean)*d.projected_gravity_b[:,2]).clamp(-1,1)
        return torch.exp(-torch.acos(cos)**2/(.08 if env.cfg.sunday_revision>=8 else (.12 if env.cfg.sunday_revision>=5 else .25))**2)
    if kind=='height':return torch.exp(-((height(env)-env.cfg.sunday_h0)/(.08*env.cfg.sunday_h0))**2)
    if kind=='posture':
        sigma=torch.where(active,.35,.12)[:,None]
        return torch.exp(-torch.mean(((d.joint_pos[:,ids]-d.default_joint_pos[:,ids])/sigma)**2,dim=1))
    if kind=='alive':return torch.ones(env.num_envs,device=env.device)
    if kind=='still':
        quiet=torch.exp(-torch.mean((d.joint_vel[:,ids]/act.speeds)**2,dim=1)-torch.sum((d.root_link_lin_vel_b[:,:2]/.05)**2,dim=1))
        return quiet*(~active)*contact(env).all(dim=1)
    if kind=='action_rate':return torch.mean((env.action_manager.action-env.action_manager.prev_action)**2,dim=1)
    if kind=='torque':return torch.mean((d.actuator_force/act.torque_limits)**2,dim=1)
    if kind=='vz':return (d.root_link_lin_vel_w[:,2]/.30)**2
    if kind=='omega_xy':return torch.sum(d.root_link_ang_vel_b[:,:2]**2,dim=1)
    if kind=='soft_limits':
        lim=d.soft_joint_pos_limits[:,ids]
        overflow=(lim[...,0]-d.joint_pos[:,ids]).clamp(min=0)+(d.joint_pos[:,ids]-lim[...,1]).clamp(min=0)
        return torch.mean((overflow/((lim[...,1]-lim[...,0])*.1).clamp(min=.01))**2,dim=1)
    if kind in ('slip','clearance'):
        sid=env.scene['robot'].find_sites(('l_sole','r_sole'),preserve_order=True)[0]
        if kind=='slip':
            return (contact(env)*torch.sum((d.site_lin_vel_w[:,sid,:2]/.1)**2,dim=-1)).mean(dim=1)
        z=d.site_pos_w[:,sid,2]-env.scene.env_origins[:,None,2]
        if env.cfg.sunday_revision>=2:
            _,z=sole_geometry(env)
            p=(env.episode_length_buf*env.step_dt)%1
            foot_phase=torch.stack([(p-.6)/.35,(p-.1)/.35],dim=1)
            swing=(foot_phase>0)&(foot_phase<1)
            target=.012*torch.sin(torch.pi*foot_phase.clamp(0,1))*swing
            # Smooth shaping over an explicit schedule; no free reward for force flicker.
            return torch.exp(-((z-target)/.008)**2).mean(dim=1)*active
        swing=(~contact(env)) & active[:,None]
        return (swing*((z-.025)/.02)**2).sum(dim=1)/swing.sum(dim=1).clamp(min=1)
    if kind=='flight':return (~contact(env)).all(dim=1).float()
    if kind=='ground':return (env.scene['body_ground'].data.found>0).any(dim=1).float()
    if kind=='self':return (env.scene['self_collision'].data.found>0).any(dim=1).float()
    if kind=='phase':
        period=1. if env.cfg.sunday_revision>=2 else .75
        p=(env.episode_length_buf*env.step_dt/period)%1
        # Double-support windows around phase boundaries, otherwise one swing foot.
        desired=torch.stack([(p<.6)|(p>.95),(p>.45)|(p<.1)],dim=1)
        if env.cfg.sunday_revision>=3:
            _,z=sole_geometry(env)
            swing=~desired
            support=(contact(env)|swing).all(dim=1)
            quality=((z/.010).clamp(0,1)*swing*(~contact(env))).sum(dim=1)
            return torch.where(swing.any(dim=1),quality*support,.25*contact(env).all(dim=1))*active
        return (contact(env)==desired).float().mean(dim=1)*active
    raise ValueError(kind)


class SustainedFailure:
    def __init__(self,cfg,env):
        self.count=torch.zeros(env.num_envs,device=env.device,dtype=torch.int32)

    def reset(self,env_ids):self.count[env_ids]=0

    def __call__(self,env,kind):
        d=env.scene['robot'].data
        if kind=='tilt':
            cos=-d.projected_gravity_b[:,2]
            bad=cos<.5
            immediate=cos<.173648
            limit=5
        elif kind=='height':bad=height(env)<.55*env.cfg.sunday_h0;immediate=torch.zeros_like(bad);limit=5
        elif kind=='ground':
            f=env.scene['body_ground'].data.force
            bad=(torch.linalg.vector_norm(f,dim=-1)>.02*env.cfg.sunday_mass*9.81).any(dim=1)
            immediate=torch.zeros_like(bad);limit=2
        elif kind=='self':bad=(env.scene['self_collision'].data.found>0).any(dim=1);immediate=torch.zeros_like(bad);limit=2
        else:raise ValueError(kind)
        self.count[:]=torch.where(bad,self.count+1,0)
        return (self.count>=limit)|immediate


def reset_nominal_subset(env,env_ids):
    """Expose the exact reference start during training alongside random starts."""
    from mjlab.envs import mdp as basic
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    if not hasattr(env,'_nominal_start_mask'):
        env._nominal_start_mask=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    selected=env_ids[torch.rand(len(env_ids),device=env.device)<env.cfg.sunday_nominal_start_fraction]
    env._nominal_start_mask[env_ids]=False
    env._nominal_start_mask[selected]=True
    if len(selected):basic.reset_scene_to_default(env,selected)


def finite_failure(env):
    d=env.scene['robot'].data
    bad=(~torch.isfinite(d.joint_pos)).any(dim=1)|(~torch.isfinite(d.joint_vel)).any(dim=1)|(~torch.isfinite(d.root_link_pos_w)).any(dim=1)
    if bad.any():
        raise FloatingPointError('Nonfinite simulation state: run stopped, not silently used in PPO')
    return bad


def boundary(env):
    return torch.linalg.vector_norm(env.scene['robot'].data.root_link_pos_w[:,:2]-env.scene.env_origins[:,:2],dim=1)>5

