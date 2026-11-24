"""Static kinematic lift feasibility within unchanged policy/joint bounds.

This is not a dynamic balance, torque, speed or hardware walking test.
"""

import bootstrap
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sunday_locomotion.env_cfg import robot_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Knee/ankle-roll residual gain inside the same mechanical soft limits",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = bootstrap.BASE / "policies/joystick_walking_r25_300/dynamics"
    home = json.loads((directory / "arm_home.json").read_text())
    model = robot_spec(directory / "sunday_a1_locomotion.xml").compile()
    data = mujoco.MjData(model)
    data.qpos[2] = home["root_height"]
    for name, value in home["joint_pos"].items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    records = []
    for side in ("l", "r"):
        names = [name for name in home["leg_order"] if name.startswith(side + "_")]
        joint_ids = np.array([model.joint(name).id for name in names])
        addresses = model.jnt_qposadr[joint_ids]
        default = initial[addresses]
        scales = np.array(
            [
                0.65 if "knee" in name else (0.45 if "roll" in name else 0.35)
                for name in names
            ]
        )
        scales *= np.array(
            [
                args.gain if ("knee" in name or "ankle_roll" in name) else 1.0
                for name in names
            ]
        )
        ranges = model.jnt_range[joint_ids]
        centers = ranges.mean(axis=1)
        half = (ranges[:, 1] - ranges[:, 0]) * home["soft_joint_pos_limit_factor"] / 2
        lower = np.maximum(default - scales, centers - half)
        upper = np.minimum(default + scales, centers + half)
        site = model.site(side + "_sole").id
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)
        origin = data.site_xpos[site].copy()
        orientation = data.site_xmat[site].reshape(3, 3).copy()
        tilt_allowance = abs(orientation[2, 0]) * 0.0525 + abs(orientation[2, 1]) * 0.09
        for forward_offset in (-0.03, 0.0, 0.03):
            guess = np.clip(default, lower + 1e-8, upper - 1e-8)
            for lift in (0.030, 0.035, 0.040, 0.045):
                target = origin + [forward_offset, 0.0, 0.0]
                target[2] = lift + tilt_allowance

                def residual(values):
                    data.qpos[:] = initial
                    data.qpos[addresses] = values
                    mujoco.mj_forward(model, data)
                    actual_rotation = data.site_xmat[site].reshape(3, 3)
                    angle = Rotation.from_matrix(
                        orientation.T @ actual_rotation
                    ).as_rotvec()
                    return np.concatenate(
                        (
                            (data.site_xpos[site] - target) / 0.005,
                            angle / 0.05,
                            0.01 * (values - default) / scales,
                        )
                    )

                solved = least_squares(
                    residual,
                    guess,
                    bounds=(lower, upper),
                    max_nfev=1000,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                residual(solved.x)
                rotation = data.site_xmat[site].reshape(3, 3)
                lowest = (
                    data.site_xpos[site, 2]
                    - abs(rotation[2, 0]) * 0.0525
                    - abs(rotation[2, 1]) * 0.09
                )
                position_error = float(np.linalg.norm(data.site_xpos[site] - target))
                angle_error = float(
                    np.linalg.norm(
                        Rotation.from_matrix(orientation.T @ rotation).as_rotvec()
                    )
                )
                records.append(
                    dict(
                        side=side,
                        forward_offset_m=forward_offset,
                        target_whole_sole_clearance_m=lift,
                        achieved_whole_sole_clearance_m=float(lowest),
                        position_error_m=position_error,
                        orientation_error_rad=angle_error,
                        within_policy_and_soft_bounds=bool(
                            np.all(solved.x >= lower) and np.all(solved.x <= upper)
                        ),
                        geometric_feasible=bool(
                            position_error < 0.001
                            and angle_error < 0.03
                            and lowest >= lift - 0.001
                        ),
                        joint_names=names,
                        joint_position_rad=solved.x.tolist(),
                        normalized_residual_action=(
                            (solved.x - default) / scales
                        ).tolist(),
                    )
                )
                guess = solved.x
    report = dict(
        knee_ankle_residual_gain=args.gain,
        model="Archived R25/Week7 model with unchanged mechanical limits",
        root_height_m=home["root_height"],
        cases=records,
        scope="Local least-squares single-foot inverse kinematics; an unsolved case is not a global infeasibility proof. Fixed root and opposite leg at Home. No motor rate/torque, collision-free path, COM/support or dynamic balance claim; this is not walking acceptance.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WEEK7_KINEMATIC_CHECK " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
