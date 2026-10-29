"""Fixed-base pose-hold and four-finger FlexiTac integration tasks."""
import json
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

ASSETS = Path(__file__).parent / "assets"
TASK_ID = "SundayA1-FixedBase-JointCheck-v0"


def robot_spec():
    return mujoco.MjSpec.from_file(str(ASSETS / "sunday_a1.xml"))


def make_env_cfg(num_envs=4):
    info = json.loads((ASSETS / "model_info.json").read_text(encoding="utf-8"))
    fingers = tuple(n for n in info["joint_names"] if "finger_slide" in n)
    others = tuple(n for n in info["joint_names"] if n not in fingers)
    robot = EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, info["base_height"]),
            joint_pos={n: 0.015 if n in fingers else 0.0 for n in info["joint_names"]},
        ),
        articulation=EntityArticulationInfoCfg(actuators=(
            BuiltinPositionActuatorCfg(
                target_names_expr=others, stiffness=100.0, damping=10.0,
                effort_limit=100.0, armature=0.01,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=fingers, stiffness=1000.0, damping=30.0,
                effort_limit=100.0, armature=0.001,
            ),
        )),
    )
    terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    scales = {n: 0.015 if n in fingers else 0.1 for n in info["joint_names"]}
    # Terrain must accept robot collision bit 1; robot accepts terrain bit 2.
    def configure_floor(spec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                geom.contype = 2
                geom.conaffinity = 1

    return ManagerBasedRlEnvCfg(
        seed=42, decimation=10, episode_length_s=10.0,
        scene=SceneCfg(
            num_envs=num_envs, env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot}, spec_fn=configure_floor,
        ),
        sim=SimulationCfg(
            nconmax=64, njmax=256,
            mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=10),
        ),
        observations={
            "actor": ObservationGroupCfg(terms, enable_corruption=False),
            "critic": ObservationGroupCfg(dict(terms), enable_corruption=False),
        },
        actions={"joint_position": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale=scales, use_default_offset=True, clip={".*": (-1.0, 1.0)},
        )},
        events={
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform, mode="reset",
                params={"pose_range": {}, "velocity_range": {}},
            ),
            "reset_joints": EventTermCfg(
                func=mdp.reset_joints_by_offset, mode="reset",
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
        rewards={
            "pose": RewardTermCfg(func=mdp.posture, weight=1.0, params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "std": {n: 0.015 if n in fingers else 0.2 for n in info["joint_names"]},
            }),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001),
        },
        terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="base_link", distance=2.2,
        ),
    )


register_mjlab_task(
    task_id=TASK_ID, env_cfg=make_env_cfg(), play_env_cfg=make_env_cfg(1),
    rl_cfg=RslRlOnPolicyRunnerCfg(
        experiment_name="sunday_a1_joint_check", logger="tensorboard",
        upload_model=False, num_steps_per_env=24, max_iterations=100,
        save_interval=50, clip_actions=1.0,
    ),
)


def make_tactile_env_cfg(num_envs=4):
    from .mjlab_flexitac import FlexiTacSensorCfg, tactile_observation

    cfg = make_env_cfg(num_envs)
    cfg.scene.sensors = (FlexiTacSensorCfg(name="flexitac"),)
    for group in cfg.observations.values():
        group.terms["flexitac"] = ObservationTermCfg(func=tactile_observation)
    return cfg


register_mjlab_task(
    task_id="SundayA1-FixedBase-FlexiTac-v0",
    env_cfg=make_tactile_env_cfg(), play_env_cfg=make_tactile_env_cfg(1),
    rl_cfg=RslRlOnPolicyRunnerCfg(
        experiment_name="sunday_a1_flexitac", logger="tensorboard", upload_model=False,
        clip_actions=1.0, num_steps_per_env=24, max_iterations=100,
    ),
)

from .ball_grasp import TASK_ID as BALL_TASK_ID, make_ball_env_cfg

register_mjlab_task(
    task_id=BALL_TASK_ID, env_cfg=make_ball_env_cfg(4), play_env_cfg=make_ball_env_cfg(1),
    rl_cfg=RslRlOnPolicyRunnerCfg(
        experiment_name="sunday_a1_ball_grasp", logger="tensorboard", upload_model=False,
        clip_actions=1.0, num_steps_per_env=24, max_iterations=100,
    ),
)
