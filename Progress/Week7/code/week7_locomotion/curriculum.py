"""Explicit training targets, distinct from achieved or accepted performance."""

import math

ARM_NAMES = tuple(
    side + "_" + suffix
    for side in ("r", "l")
    for suffix in (
        "shoulder_roll",
        "shoulder_pitch",
        "elbow_yaw",
        "elbow_pitch",
        "wrist_yaw",
    )
)
ACTOR_INPUTS = 510
CRITIC_INPUTS = 108
ACTION_OUTPUTS = 22
TASK_VERSION = "sunday.week7.r26.arm_reference.v1"


def targets(progress, training_round=1):
    if not math.isfinite(progress):
        raise ValueError("Curriculum progress must be finite")
    if training_round not in (1, 2, 3, 4):
        raise ValueError("Training round must be 1, 2, 3 or 4")
    p = min(1.0, max(0.0, progress))
    second = training_round >= 2
    return {
        "progress": p,
        "forward_cap_m_s": (
            0.20
            if training_round >= 3
            else ((0.18 + 0.02 * p) if second else (0.12 + 0.08 * p))
        ),
        "reverse_cap_m_s": 0.06,
        "yaw_cap_rad_s": 0.25,
        "swing_target_m": (
            0.045
            if training_round >= 3
            else ((0.035 + 0.010 * p) if second else (0.024 + 0.011 * p))
        ),
        "arm_fraction": 1.0 if second else 0.10 + 0.90 * p,
        "lift_cap_rad": 0.70 * (1.0 if second else 0.10 + 0.90 * p),
        "spread_cap_rad": 0.65 * (1.0 if second else 0.10 + 0.90 * p),
    }


def progress(env):
    fixed = env.cfg.week7_progress_override
    if fixed is not None:
        return float(fixed)
    return min(1.0, env.common_step_counter / (32 * env.cfg.week7_ramp_iterations))


def env_targets(env):
    return targets(progress(env), env.cfg.week7_training_round)
