"""Flat-ground task using a fully floating robot and no tactile PPO overhead."""
from dataclasses import dataclass
import json
from functools import partial
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp as basic
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.noise import UniformNoiseCfg

from .paths import MODEL, HOME
from . import mdp


@dataclass(kw_only=True)
class SundayEnvCfg(ManagerBasedRlEnvCfg):
    sunday_h0: float=.247
    sunday_mass: float=2.84
    sunday_stage: str='stand'
    sunday_whole_body: bool=False
    sunday_phase: bool=False
    sunday_revision: int=1
    sunday_nominal_start_fraction: float=0.
    sunday_action_gain_curriculum: bool=False
    sunday_cadence_curriculum: bool=False
    sunday_sole_xy: tuple=((-0.004,.07),(-0.004,-.07))


def robot_spec(model_path=None):
    path=Path(model_path) if model_path is not None else MODEL
    if path==MODEL:return mujoco.MjSpec.from_file(str(path))
    # Archived run XML uses the same frozen meshes, resolved without copying them.
    root=ET.parse(path).getroot()
    root.find('compiler').set('meshdir',str(MODEL.parent/'meshes'))
    return mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'))


def make_env(num_envs=64,stage='stand',whole_body=False,play=False,gait_phase=False,revision=1,nominal_path=None,model_path=None,nominal_start_fraction=0.):
    if nominal_path is None and revision>=24:nominal_path=HOME.with_name('arm_clearance_home.json')
    home=json.loads((nominal_path if nominal_path is not None else (HOME.with_name('arm_home.json') if revision>=7 else (HOME.with_name('natural_home.json') if revision>=6 else (HOME.with_name('quality_home.json') if revision>=5 else (HOME.with_name('gait_home.json') if revision>=2 else HOME))))).read_text())
    if revision>=2:gait_phase=True
    names=tuple(n for n in home['joint_order'] if n in home['leg_order'] or (whole_body and n in home['arm_order']))
    robot=EntityCfg(spec_fn=partial(robot_spec,model_path),
        init_state=EntityCfg.InitialStateCfg(pos=(0,0,home['root_height']),joint_pos=home['joint_pos']),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=tuple(home['joint_order'])),),soft_joint_pos_limit_factor=home.get('soft_joint_pos_limit_factor',.9)))
    actor={
        'gyro':ObservationTermCfg(func=basic.base_ang_vel,scale=.25,noise=UniformNoiseCfg(n_min=-.03,n_max=.03)),
        'gravity':ObservationTermCfg(func=basic.projected_gravity),
        'command':ObservationTermCfg(func=mdp.command),
        'joint_pos':ObservationTermCfg(func=basic.joint_pos_rel),
        'joint_vel':ObservationTermCfg(func=basic.joint_vel_rel,scale=.05),
        'last_action':ObservationTermCfg(func=basic.last_action),
    }
    if gait_phase:actor['phase']=ObservationTermCfg(func=mdp.phase)
    if revision>=4:actor['heading']=ObservationTermCfg(func=mdp.heading)
    critic={**actor,'linear_velocity':ObservationTermCfg(func=basic.base_lin_vel),'feet':ObservationTermCfg(func=mdp.foot_state)}
    if revision>=10:
        # Simulated base velocity; a real robot will need a velocity estimator.
        # Append after constructing critic so its established 98-dim schema stays intact.
        actor['velocity_feedback']=ObservationTermCfg(func=basic.base_lin_vel,noise=UniformNoiseCfg(n_min=-.01,n_max=.01))
    sensors=(
        ContactSensorCfg(name='feet_contact',primary=ContactMatch(mode='body',pattern=r'^[lr]_foot_link_1$',entity='robot'),
                         secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',track_air_time=True,history_length=10 if revision>=2 else 0),
        ContactSensorCfg(name='body_ground',primary=ContactMatch(mode='body',pattern=r'^(?![lr]_foot_link_1$|base_link$).*',entity='robot'),
                         secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce'),
        ContactSensorCfg(name='self_collision',primary=ContactMatch(mode='subtree',pattern='base_link',entity='robot'),
                         secondary=ContactMatch(mode='subtree',pattern='base_link',entity='robot'),fields=('found',),reduce='none'),
    )
    weights={'track_xy':2.,'track_yaw':1.,'upright':.8,'height':.4,'posture':.2,'alive':.1,'still':.3,
             'action_rate':-.05,'torque':-.05,'vz':-.1,'omega_xy':-.05,'slip':-.2,'clearance':-.1,
             'soft_limits':-1.,'ground':-1.,'self':-.5,'flight':-.3}
    if gait_phase:weights['phase']=.2
    if revision>=2:
        weights.update(track_xy=3.,progress=4.,under_speed=-1.,track_yaw=.3,upright=.5,height=.25,
                       posture=.05,alive=.02,still=.2,action_rate=-.02,torque=-.01,
                       vz=-.05,omega_xy=-.03,slip=-.05,clearance=1.,phase=.7,flight=-.5,failure=-3.)
    if revision>=3:
        weights.update(track_xy=4.,progress=2.,velocity_cost=-.5,yaw_cost=-.2,track_yaw=.5,
                       upright=.7,slip=-.5,clearance=2.,phase=2.,action_rate=-.1,omega_xy=-.1)
    if revision>=4:weights.update(heading_cost=-.5,lateral_speed=-.4)
    if revision>=5:
        for name in ('lateral_speed','clearance','phase','slip'):weights.pop(name,None)
        weights.update(upright=2.,height=.5,omega_xy=-.4,heading_cost=-1.,
                       quality_clearance=2.,quality_foot_path=1.5,quality_sway=.6,
                       quality_plant_slip=-.4,quality_foot_flat=-.3,quality_swing_contact=-.8,
                       quality_impact=-.4,quality_joint_acc=-.15,quality_phase=1.)
    if revision>=7:
        weights.update(quality_phase=3.,quality_swing_contact=-1.5,quality_plant_slip=-.6)
        if whole_body:weights['quality_arms']=.4
    if revision>=8:
        weights.update(upright=3.,omega_xy=-.8,quality_plant_slip=-.8,quality_landing_brake=-.35)
    if revision>=9:weights['quality_lane']=-.6
    if revision>=11:weights.pop('quality_lane',None)
    if revision>=12:
        weights.update(track_yaw=2.,yaw_cost=-.5,heading_cost=-.5,still=1.,posture=.15,
                       quality_stop=2.5,quality_stop_cost=-1.,quality_foot_heading=.3)
    if revision>=13:
        weights.update(quality_stop=4.,quality_stop_cost=-1.5,quality_stop_feet=-1.5,quality_stop_pose=1.2,still=1.5)
    if revision>=14:
        weights.update(track_xy=6.,velocity_cost=-1.,quality_clearance=3.,quality_clearance_margin=-1.5,
                       quality_phase=4.,quality_swing_contact=-3.,quality_plant_slip=-1.2,quality_foot_heading=.5)
    if revision>=15:
        weights.update(yaw_cost=-1.5,quality_transfer_cost=-.8,quality_clearance_margin=-3.)
    if revision>=16:weights.update(quality_stop_transition=-.75,quality_in_place_cost=-1.5)
    if revision>=18:weights['quality_load_transfer']=-1.
    if revision>=20:weights.update(quality_hand_clearance=-1.,quality_torso_oscillation=-.6,quality_in_place_cost=-2.5)
    if revision>=21:weights.update(quality_hand_clearance=-2.,quality_torso_oscillation=-1.)
    if revision>=23:weights.update(velocity_cost=-2.,yaw_cost=-2.,quality_hand_clearance=-3.,quality_foot_separation=-1.5,quality_torso_oscillation=-1.25)
    if revision>=24:weights.update(quality_torso_oscillation=-2.,quality_foot_separation=-2.)
    rewards={k:RewardTermCfg(func=mdp.reward,weight=v,params={'kind':k}) for k,v in weights.items()}
    if stage=='stand':rewards.pop('clearance',None)
    reset_root=EventTermCfg(func=basic.reset_root_state_uniform,mode='reset',params={
        'pose_range':{} if play else {'roll':(-.015,.015),'pitch':(-.015,.015)},'velocity_range':{}})
    reset_joint=EventTermCfg(func=basic.reset_joints_by_offset,mode='reset',params={
        'position_range':(0,0) if play else (-.01,.01),'velocity_range':(0,0),
        'asset_cfg':SceneEntityCfg('robot',joint_names=names)})
    reset_events={'defaults':EventTermCfg(func=basic.reset_scene_to_default,mode='reset'),
                  'reset_root':reset_root,'reset_joints':reset_joint}
    if nominal_start_fraction:
        if not 0<nominal_start_fraction<1:raise ValueError('Nominal fraction must preserve randomized starts')
        reset_events['nominal_subset']=EventTermCfg(func=mdp.reset_nominal_subset,mode='reset')
    return SundayEnvCfg(seed=42,scene=SceneCfg(num_envs=num_envs,env_spacing=2.,terrain=TerrainEntityCfg(terrain_type='plane'),
            entities={'robot':robot},sensors=sensors),
        sim=SimulationCfg(nconmax=128,njmax=512,contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(timestep=.002,iterations=20,ls_iterations=10,cone='elliptic',integrator='implicitfast')),
        decimation=10,episode_length_s=10. if stage=='stand' else ((30. if revision>=13 else 24.) if stage=='control' else 20.),
        observations={'actor':ObservationGroupCfg(terms=actor,history_length=5,enable_corruption=not play,nan_policy='error'),
                      'critic':ObservationGroupCfg(terms=critic,enable_corruption=False,nan_policy='error')},
        actions={'residual':mdp.ResidualActionCfg(entity_name='robot',joint_names=names)},
        commands={'twist':mdp.TwistCfg(resampling_time_range=((3.,9.) if revision>=21 else ((6.,9.) if revision>=13 else (3.,5.))) if stage=='control' else (2.,5.),stage=stage)},
        events=reset_events,rewards=rewards,
        terminations={'time_out':TerminationTermCfg(func=basic.time_out,time_out=True),
                      **{k:TerminationTermCfg(func=mdp.SustainedFailure,params={'kind':k}) for k in ('tilt','height','ground','self')},
                      'nonfinite':TerminationTermCfg(func=mdp.finite_failure),
                      'boundary':TerminationTermCfg(func=mdp.boundary,time_out=True)},
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,entity_name='robot',body_name='base_link',distance=1.6,elevation=-15.,azimuth=135.),
        sunday_h0=home['root_height'],sunday_mass=home['total_mass_kg'],sunday_stage=stage,sunday_whole_body=whole_body,sunday_phase=gait_phase,sunday_revision=revision,sunday_nominal_start_fraction=nominal_start_fraction,sunday_sole_xy=tuple(tuple(x) for x in home.get('sole_center_xy_root',((-0.004,.07),(-0.004,-.07)))))


def agent_cfg(revision=1):
    from mjlab.rl import RslRlOnPolicyRunnerCfg,RslRlModelCfg,RslRlPpoAlgorithmCfg
    algorithm_class=RslRlPpoAlgorithmCfg;algorithm_extra={}
    if revision>=17:
        from .learning_cfg import MirrorPpoCfg
        algorithm_class=MirrorPpoCfg
        algorithm_extra['symmetry_cfg']=dict(use_data_augmentation=False,use_mirror_loss=True,mirror_loss_coeff=.5,
            data_augmentation_func='sunday_locomotion.symmetry:augment')
    return RslRlOnPolicyRunnerCfg(actor=RslRlModelCfg(hidden_dims=(256,128,128),obs_normalization=True,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.25 if revision>=7 else (.15 if revision>=6 else (.22 if revision>=2 else .35)),'std_type':'scalar',
                              **({'std_range':(.03 if revision>=16 else (.02 if revision>=13 else (.03 if revision>=12 else .05)),.5)} if revision>=7 else {})}),
        critic=RslRlModelCfg(hidden_dims=(256,128,128),obs_normalization=True),
        algorithm=algorithm_class(learning_rate=1.5e-4 if revision>=21 else 3e-4,entropy_coef=.0005 if revision>=13 else (.002 if revision>=8 else (.005 if revision>=7 else (.001 if revision>=2 else .01))),**algorithm_extra),
        num_steps_per_env=32 if revision>=2 else 24,logger='tensorboard',upload_model=False,save_interval=25,clip_actions=1.)
