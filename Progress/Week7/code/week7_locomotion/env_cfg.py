"""R26 overlays the archived R25 model with an explicit new observation contract."""

from dataclasses import dataclass, fields
from pathlib import Path

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from sunday_locomotion.env_cfg import (
    SundayEnvCfg,
    make_env as make_base,
    agent_cfg as base_agent,
)
from . import mdp

BASE = Path(__file__).resolve().parents[3] / "Week4"
DYNAMICS = BASE / "policies/joystick_walking_r25_300/dynamics"


@dataclass(kw_only=True)
class Week7EnvCfg(SundayEnvCfg):
    week7_randomize_phase: bool = False
    week7_training_round: int = 1
    week7_ramp_iterations: int = 600
    week7_progress_override: float | None = None


def make_env(
    num_envs=256,
    seed=742,
    play=False,
    progress_override=None,
    ramp_iterations=600,
    training_round=1,
):
    original = make_base(
        num_envs,
        "control",
        True,
        play=play,
        gait_phase=True,
        revision=25,
        nominal_path=DYNAMICS / "arm_home.json",
        model_path=DYNAMICS / "sunday_a1_locomotion.xml",
    )
    cfg = Week7EnvCfg(**{f.name: getattr(original, f.name) for f in fields(original)})
    if training_round not in (1, 2, 3, 4):
        raise ValueError("Training round must be 1, 2, 3 or 4")
    cfg.week7_training_round = training_round
    cfg.week7_randomize_phase = training_round >= 3 and not play
    cfg.seed = seed
    cfg.sunday_revision = 26
    cfg.week7_progress_override = progress_override
    cfg.week7_ramp_iterations = ramp_iterations
    names = cfg.actions["residual"].joint_names
    cfg.actions["residual"] = mdp.HeldArmActionCfg(
        entity_name="robot", joint_names=names
    )
    cfg.commands["twist"] = mdp.Week7TwistCfg(
        stage="control", resampling_time_range=(3.0, 8.0)
    )
    cfg.commands["arms"] = mdp.ArmReferenceCfg(resampling_time_range=(1.0e9, 1.0e9))
    # Append to preserve the entire term-major R25 prefix for weight migration.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["arm_reference"] = ObservationTermCfg(
            func=mdp.arm_reference
        )
    cfg.rewards["quality_clearance"] = RewardTermCfg(
        func=mdp.clearance_reward, weight=4.0
    )
    cfg.rewards["quality_clearance_margin"] = RewardTermCfg(
        func=mdp.clearance_margin_cost, weight=-3.0
    )
    cfg.rewards["posture"] = RewardTermCfg(func=mdp.leg_posture, weight=0.15)
    cfg.rewards["quality_stop_pose"] = RewardTermCfg(
        func=mdp.leg_posture, weight=1.2, params={"stopped": True}
    )
    cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.leg_action_rate, weight=-0.02)
    cfg.rewards.pop("quality_arms", None)
    cfg.rewards["arm_tracking"] = RewardTermCfg(func=mdp.arm_tracking, weight=0.3)
    if training_round >= 2:
        cfg.rewards["quality_clearance"].weight = 5.0
        cfg.rewards["quality_clearance_margin"].weight = -4.0
        cfg.rewards["quality_foot_heading"].weight = 1.0
        cfg.rewards["foot_heading_cost"] = RewardTermCfg(
            func=mdp.foot_heading_cost, weight=-2.5
        )
        cfg.rewards["toe_in_cost"] = RewardTermCfg(func=mdp.toe_in_cost, weight=-1.5)
        cfg.rewards["quality_foot_separation"].weight = -3.0
        cfg.rewards["quality_hand_clearance"].weight = -4.5
    if training_round >= 3:
        from .balance import PhaseMatchedClearanceCost

        cfg.rewards["phase_matched_clearance"] = RewardTermCfg(
            func=PhaseMatchedClearanceCost, weight=-1.0
        )
        cfg.rewards["toe_in_cost"].weight = -2.0
    if training_round >= 4:
        from .balance import LateralBiasCost

        cfg.rewards["quality_clearance"].weight = 6.0
        cfg.rewards["quality_clearance_margin"].weight = -5.0
        cfg.rewards["phase_matched_clearance"].weight = -0.35
        cfg.rewards["foot_heading_cost"].weight = -3.5
        cfg.rewards["toe_in_cost"].weight = -3.0
        cfg.rewards["lateral_bias"] = RewardTermCfg(func=LateralBiasCost, weight=-0.75)
        for name in ("track_xy", "progress", "track_yaw"):
            cfg.rewards[name].func = mdp.lift_gated_base_reward
    return cfg


def agent_cfg(training_round=1):
    cfg = base_agent(25)
    # The old mirror map assumes 460/98 features and fixed arm references.
    cfg.algorithm.symmetry_cfg = None
    if training_round >= 3:
        cfg.algorithm.symmetry_cfg = dict(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.05 if training_round >= 4 else 0.20,
            data_augmentation_func="week7_locomotion.symmetry:augment",
        )
    cfg.algorithm.learning_rate = 5e-5 if training_round >= 4 else 7.5e-5
    cfg.algorithm.entropy_coef = 0.0005
    cfg.save_interval = 25
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
