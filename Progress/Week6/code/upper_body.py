"""Rate-limited bilateral joint targets for the archived Sunday A1 simulation.

Joint names are CAD conventions: raising uses mirrored shoulder_roll, while
spreading uses positive shoulder_pitch on BOTH sides. Geometry tests establish
these signs; do not infer the mapping from the English joint names.
"""

import math

import mujoco

from controller import axis

ARM_SPEED = 0.65  # rad/s; diagonal stick input shares this angular budget
FINGER_SPEED = 0.024  # m/s per slider
LIFT_LIMIT = 0.70  # rad above the saved Home
SPREAD_LIMIT = 0.65  # rad beyond the saved Home
ARM_SUFFIXES = ("shoulder_roll", "shoulder_pitch", "elbow_yaw", "elbow_pitch", "wrist_yaw")
ARM_NAMES = tuple(s + "_" + j for s in ("l", "r") for j in ARM_SUFFIXES)
FINGER_NAMES = tuple(s + "_finger_slide_" + f for s in ("l", "r") for f in ("top", "bottom"))


class UpperBody:
    def __init__(self, model, home):
        self.home = {name: float(home[name]) for name in ARM_NAMES + FINGER_NAMES}
        self.bounds = {}
        self.qadr = {}
        for name in self.home:
            # Tests compile the same XML alone; mjlab adds the robot/ namespace.
            prefix = (
                "robot/"
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + name) >= 0
                else ""
            )
            joint = model.joint(prefix + name)
            if joint.id < 0 or not model.jnt_limited[joint.id]:
                raise ValueError("Missing/unsupported limited joint: " + name)
            actuator = model.actuator(prefix + name)
            if actuator.id < 0 or model.actuator_trnid[actuator.id, 0] != joint.id:
                raise ValueError("Missing/mismatched position actuator: " + name)
            low, high = map(float, joint.range)
            margin = 0.05 * (high - low)  # Week4's 90% soft position envelope
            self.bounds[name] = (low + margin, high - margin)
            self.qadr[name] = int(joint.qposadr[0])
            if not self.bounds[name][0] <= self.home[name] <= self.bounds[name][1]:
                raise ValueError("Home outside soft limits: " + name)
        # Symmetric limits are derived from BOTH mirrored arm joint ranges.
        self.lift_max = min(
            LIFT_LIMIT,
            self.home["l_shoulder_roll"] - self.bounds["l_shoulder_roll"][0],
            self.bounds["r_shoulder_roll"][1] - self.home["r_shoulder_roll"],
        )
        self.spread_max = min(
            SPREAD_LIMIT,
            *(
                self.bounds[s + "_shoulder_pitch"][1] - self.home[s + "_shoulder_pitch"]
                for s in ("l", "r")
            ),
        )
        self.open_position = min(self.bounds[n][1] for n in FINGER_NAMES)
        self.closed_position = max(self.bounds[n][0] for n in FINGER_NAMES)
        self.reset()

    def reset(self):
        self.lift = self.spread = 0.0
        self.positions = dict(self.home)
        self.grip_goal = self.home[FINGER_NAMES[0]]
        self.grip_state = "Home aperture"
        self.returning = False

    def at_home(self):
        return all(abs(self.positions[n] - v) < 1e-8 for n, v in self.home.items())

    def apply(self, targets, names, state, dt):
        """Override only upper-body targets, once per simulation control step."""
        if not math.isfinite(dt) or not 0 < dt <= 0.05:
            raise ValueError("Upper-body step must be in (0, 0.05] seconds")
        if state.get("upper_enabled", False):
            if state.get("home", False):
                self.returning = True
            if self.returning:
                self.lift = max(0.0, self.lift - ARM_SPEED * dt)
                self.spread = max(0.0, self.spread - ARM_SPEED * dt)
                self.grip_goal = self.home[FINGER_NAMES[0]]
                self.grip_state = "Returning arms and hands to Home"
            else:
                lift, spread = -axis(state["ry"]), axis(state["rx"])
                length = max(1.0, math.hypot(lift, spread))
                self.lift = min(self.lift_max, max(0.0, self.lift + lift / length * ARM_SPEED * dt))
                self.spread = min(
                    self.spread_max, max(0.0, self.spread + spread / length * ARM_SPEED * dt)
                )
                a, y = state["a"], state["y_button"]
                if a and y:
                    self.grip_goal = self.positions[FINGER_NAMES[0]]
                    self.grip_state = "A+Y: hold"
                elif a:
                    self.grip_goal, self.grip_state = self.open_position, "Opening both hands"
                elif y:
                    self.grip_goal, self.grip_state = self.closed_position, "Closing both hands"
        else:
            # Cancel a latched hand movement on stop, disconnect, or timeout.
            self.grip_goal = self.positions[FINGER_NAMES[0]]
            self.grip_state = "Input disabled: hold targets"
            self.returning = False
        self.positions["l_shoulder_roll"] = self.home["l_shoulder_roll"] - self.lift
        self.positions["r_shoulder_roll"] = self.home["r_shoulder_roll"] + self.lift
        for side in ("l", "r"):
            name = side + "_shoulder_pitch"
            self.positions[name] = self.home[name] + self.spread
        for name in FINGER_NAMES:
            delta = self.grip_goal - self.positions[name]
            self.positions[name] += max(-FINGER_SPEED * dt, min(FINGER_SPEED * dt, delta))
        for name, value in self.positions.items():
            low, high = self.bounds[name]
            targets[names.index(name)] = max(low, min(high, value))
        if self.returning and self.at_home():
            self.returning = False
            self.grip_state = "Home aperture"

    def status(self):
        return dict(
            lift_rad=self.lift,
            spread_rad=self.spread,
            lift_limit_rad=self.lift_max,
            spread_limit_rad=self.spread_max,
            grip_state=self.grip_state,
            slider_goal_m=self.grip_goal,
            slider_target_m=self.positions[FINGER_NAMES[0]],
            targets=dict(self.positions),
        )


class ArmTeleop:
    """Adjust arms at standstill and allow walking with any held arm pose.

    Before manipulation the original policy supplies its arm residual targets.
    After manipulation the overlay holds that pose while the original policy
    continues to supply leg actions. Home is optional, never a walking gate.
    """

    SETTLE_SECONDS = 0.60
    HOME_SECONDS = 0.30

    def __init__(self, model, data, home):
        self.model, self.data = model, data
        self.upper = UpperBody(model, home)
        self.qadr = self.upper.qadr
        prefix = (
            "robot/"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link") >= 0
            else ""
        )
        self.root = model.body(prefix + "base_link").id
        self.reset()

    def reset(self):
        self.upper.reset()
        self.active = self.ready = False
        self.still_time = self.home_time = 0.0
        self.reason = "Waiting for measured standstill"
        self.previous = None

    def gate(self, requested, filtered_command, dt):
        import numpy as np

        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.root, velocity, 1
        )
        neutral = max(abs(float(v)) for v in requested) < 1e-8
        command_zero = max(abs(float(v)) for v in filtered_command) < 0.005
        speed = math.hypot(velocity[3], velocity[4])
        quiet = neutral and command_zero and speed < 0.025 and abs(velocity[2]) < 0.08
        self.still_time = self.still_time + dt if quiet else 0.0
        # Hysteresis avoids cancelling a button's latched goal because the
        # arm itself produces a small, temporary body sway after acquisition.
        continuing = (
            self.active
            and self.ready
            and neutral
            and command_zero
            and speed < 0.06
            and abs(velocity[2]) < 0.16
        )
        self.ready = bool(continuing or self.still_time >= self.SETTLE_SECONDS)
        measured_home = all(
            abs(float(self.data.qpos[adr]) - self.upper.home[n])
            < (0.002 if n in FINGER_NAMES else 0.06)
            for n, adr in self.qadr.items()
        )
        home = self.upper.at_home() and measured_home
        self.home_time = self.home_time + dt if home and self.ready else 0.0
        if self.active and self.home_time >= self.HOME_SECONDS and not self.upper.returning:
            self.active = False
        if not neutral:
            self.reason = (
                "Walking with held arm pose and original leg policy"
                if self.active
                else "Walking with original policy arm references"
            )
        else:
            self.reason = (
                "Stopped: right stick / A / Y enabled"
                if self.ready
                else "Arm adjustment paused: waiting for standstill; walking available"
            )
        return list(requested)

    def apply(self, targets, names, state, dt):
        enabled = state.get("upper_enabled", False) and self.ready
        activity = (
            axis(state.get("rx", 0.0)) != 0
            or axis(state.get("ry", 0.0)) != 0
            or state.get("a", False)
            or state.get("y_button", False)
            or state.get("home", False)
        )
        if enabled and activity:
            self.active = True
            self.home_time = 0.0
        if self.active:
            self.upper.apply(targets, names, {**state, "upper_enabled": enabled}, dt)
            # Smooth the transition from the policy's small arm residuals.
            if self.previous is not None:
                for n in ARM_NAMES + FINGER_NAMES:
                    i = names.index(n)
                    speed = FINGER_SPEED if n in FINGER_NAMES else ARM_SPEED
                    targets[i] = max(
                        self.previous[n] - speed * dt,
                        min(self.previous[n] + speed * dt, targets[i]),
                    )
        self.previous = {n: float(targets[names.index(n)]) for n in ARM_NAMES + FINGER_NAMES}

    def status(self):
        return {
            **self.upper.status(),
            "stationary_ready": self.ready,
            "manipulation_active": self.active,
            "still_time_s": self.still_time,
            "home_confirm_time_s": self.home_time,
            "mode_reason": self.reason,
        }
