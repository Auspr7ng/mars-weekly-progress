"""Input failure cases and geometry checks against the archived MuJoCo model."""

import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "Week4/code/src"))

from controller import Controller, LIMITS
from upper_body import ARM_NAMES, FINGER_NAMES, UpperBody, ArmTeleop, ARM_SPEED, FINGER_SPEED
from week4_mapping import map_left_stick


class InputContract(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.seq = 0
        self.control = Controller(clock=lambda: self.now, require_lb=False, mapper=map_left_stick)

    def send(self, **kw):
        self.seq += 1
        packet = dict(
            client="test",
            seq=self.seq,
            id="test pad",
            mapping="standard",
            connected=True,
            focused=True,
            x=0.0,
            y=0.0,
            rx=0.0,
            ry=0.0,
            a=False,
            y_button=False,
            home=False,
            lb=False,
            b=False,
            menu=False,
        )
        packet.update(kw)
        return self.control.submit(packet)

    def test_neutral_requires_both_sticks_and_released_buttons(self):
        for kw in ({"rx": 1.0}, {"ry": -1.0}, {"a": True}, {"y_button": True}, {"menu": True}):
            self.control.stop()
            self.send(**kw)
            self.assertFalse(self.control.sample()["upper_enabled"])
        self.send()
        self.assertTrue(self.control.sample()["upper_enabled"])

    def test_left_stick_parity_and_right_stick_isolation(self):
        self.send()
        self.send(x=0.7, y=-0.8, rx=1.0, ry=-1.0, a=True)
        self.assertEqual(self.control.sample()["command"], map_left_stick(0.7, -0.8, LIMITS))
        self.send(rx=-1.0, ry=1.0)
        self.assertEqual(self.control.sample()["command"], [0.0, 0.0, 0.0])
        self.assertTrue(self.control.sample()["upper_enabled"])

    def test_stop_disconnect_focus_and_timeout_require_rearm(self):
        for kw in ({"b": True}, {"connected": False}, {"focused": False}, {"mapping": ""}):
            self.send()
            self.send(rx=1.0)
            self.send(rx=1.0, **kw)
            self.assertFalse(self.control.sample()["upper_enabled"])
            self.send(rx=1.0)
            self.assertFalse(self.control.sample()["upper_enabled"])
        self.send()
        self.send(a=True)
        self.now = 0.31
        self.assertFalse(self.control.sample()["upper_enabled"])
        self.send(a=True)
        self.assertFalse(self.control.sample()["upper_enabled"])

    def test_optional_lb_covers_arms_and_walking(self):
        self.control.require_lb = True
        self.send()
        self.send(rx=1.0)
        self.assertFalse(self.control.sample()["upper_enabled"])
        self.send(lb=True, rx=1.0)
        self.assertTrue(self.control.sample()["upper_enabled"])
        self.send(rx=1.0)
        self.assertFalse(self.control.sample()["upper_enabled"])

    def test_bad_packet_does_not_refresh_timestamp(self):
        self.send()
        for kw in ({"rx": float("nan")}, {"ry": float("inf")}, {"a": 1}, {"y_button": "yes"}):
            self.now = 0.1
            with self.assertRaises(ValueError):
                self.send(**kw)
            self.assertEqual(self.control.last_time, 0.0)

    def test_menu_is_one_reset_and_disarms(self):
        self.send()
        self.send(menu=True)
        self.assertTrue(self.control.take_reset())
        self.assertFalse(self.control.sample()["upper_enabled"])
        self.send(menu=True)
        self.assertFalse(self.control.take_reset())

    def test_x_requires_rearm_but_can_request_home_when_armed(self):
        self.send(home=True)
        self.assertFalse(self.control.sample()["upper_enabled"])
        self.send()
        self.send(home=True)
        self.assertTrue(self.control.sample()["upper_enabled"])
        self.assertTrue(self.control.sample()["home"])


class ModelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[2] / "Week4"
        dynamics = base / "policies/joystick_walking_r25_300/dynamics"
        root = ET.parse(dynamics / "sunday_a1_locomotion.xml").getroot()
        root.find("compiler").set("meshdir", str(base / "model/meshes"))
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        cls.home = json.loads((dynamics / "arm_home.json").read_text())["joint_pos"]
        cls.names = list(cls.home)

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.targets = np.array([self.home[n] for n in self.names])
        self.upper = UpperBody(self.model, self.home)
        self.state = dict(upper_enabled=True, rx=0.0, ry=0.0, a=False, y_button=False)

    def step(self, count=1, **kw):
        state = {**self.state, **kw}
        for _ in range(count):
            self.upper.apply(self.targets, self.names, state, 0.02)

    def geometry(self):
        for name, value in zip(self.names, self.targets):
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        return {
            side: self.data.xpos[self.model.body(side + "_gripper_base_1").id].copy()
            for side in ("l", "r")
        }

    def test_raise_and_spread_directions_in_actual_geometry(self):
        before = self.geometry()
        self.step(50, ry=-1.0)
        raised = self.geometry()
        for side in ("l", "r"):
            self.assertGreater(raised[side][2], before[side][2])
        self.step(50, ry=1.0)
        self.assertAlmostEqual(self.upper.lift, 0.0)
        self.step(50, rx=1.0)
        spread = self.geometry()
        for side in ("l", "r"):
            self.assertGreater(abs(spread[side][1]), abs(before[side][1]))
        self.step(50, rx=-1.0)
        np.testing.assert_allclose(
            list(self.geometry().values()), list(before.values()), atol=1e-12
        )

    def test_open_is_larger_physical_gap_for_both_hands(self):
        def gaps():
            self.geometry()
            return [
                np.linalg.norm(
                    self.data.geom_xpos[self.model.geom("collision_" + s + "_finger_top_1").id]
                    - self.data.geom_xpos[self.model.geom("collision_" + s + "_finger_bottom_1").id]
                )
                for s in ("l", "r")
            ]

        self.step(150, y_button=True)
        closed = gaps()
        self.step(150, a=True)
        opened = gaps()
        self.assertTrue(all(o > c + 0.04 for o, c in zip(opened, closed)))

    def test_diagonal_rate_limits_and_legs_unchanged(self):
        before = self.targets.copy()
        self.step(rx=1.0, ry=-1.0, a=True)
        self.assertLessEqual(np.hypot(self.upper.lift, self.upper.spread), ARM_SPEED * 0.02 + 1e-12)
        for name in FINGER_NAMES:
            i = self.names.index(name)
            self.assertLessEqual(abs(self.targets[i] - before[i]), FINGER_SPEED * 0.02 + 1e-12)
        for i, name in enumerate(self.names):
            if name not in ARM_NAMES + FINGER_NAMES:
                self.assertEqual(self.targets[i], before[i])

    def test_limits_hold_cancel_and_reset(self):
        self.step(500, rx=1.0, ry=-1.0, a=True)
        for name, value in self.upper.positions.items():
            low, high = self.upper.bounds[name]
            self.assertLessEqual(low, value)
            self.assertLessEqual(value, high)
        before = self.targets.copy()
        self.step(10)
        np.testing.assert_array_equal(self.targets, before)
        self.step(y_button=True)
        before = self.targets.copy()
        self.step(10, upper_enabled=False)
        np.testing.assert_array_equal(self.targets, before)
        self.step(10)  # cancelled closing must not resume on re-arm
        np.testing.assert_array_equal(self.targets, before)
        self.upper.reset()
        self.step()
        np.testing.assert_allclose(self.targets, [self.home[n] for n in self.names])

    def test_simultaneous_a_y_holds_and_press_latches(self):
        self.step(a=True)
        first = self.targets.copy()
        self.step(5)
        self.assertGreater(
            self.targets[self.names.index(FINGER_NAMES[0])],
            first[self.names.index(FINGER_NAMES[0])],
        )
        before = self.targets.copy()
        self.step(5, a=True, y_button=True)
        np.testing.assert_array_equal(before, self.targets)

    def test_x_returns_all_upper_targets_gradually(self):
        self.step(90, ry=-1.0, rx=1.0, a=True)
        before = self.upper.lift
        self.step(home=True)
        self.assertGreater(self.upper.lift, 0.0)
        self.assertLess(self.upper.lift, before)
        self.step(150)
        self.assertTrue(self.upper.at_home())
        self.assertFalse(self.upper.returning)

    def test_stationary_arm_edits_and_walking_at_nonhome_pose(self):
        self.geometry()
        control = ArmTeleop(self.model, self.data, self.home)

        def tick(request=(0.0, 0.0, 0.0), command=(0.0, 0.0, 0.0), **kw):
            result = control.gate(request, command, 0.02)
            control.apply(self.targets, self.names, {**self.state, **kw}, 0.02)
            return result

        original = self.targets.copy()
        for _ in range(20):
            tick(ry=-1.0)
        np.testing.assert_array_equal(self.targets, original)
        self.assertFalse(control.active)
        # A zero stick is insufficient while the base is still moving.
        self.data.qvel[0] = 0.1
        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):
            tick(ry=-1.0)
        self.assertFalse(control.ready)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):
            tick(command=(0.03, 0.0, 0.0), ry=-1.0)
        self.assertFalse(control.ready)
        for _ in range(40):
            tick(ry=-1.0)
        self.assertTrue(control.active)
        self.assertGreater(control.upper.lift, 0.0)
        self.data.qvel[0] = 0.03  # small arm-induced sway must not cancel a latched goal
        mujoco.mj_forward(self.model, self.data)
        tick(a=True)
        self.assertTrue(control.ready)
        json.dumps(control.status())  # viewer/trace serialization after the hysteresis path
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.geometry()  # actual joints now away from Home
        self.assertEqual(tick(request=(0.12, 0.0, 0.0)), [0.12, 0.0, 0.0])
        for _ in range(35):
            tick()
        tick(home=True)
        for _ in range(150):
            tick()
        self.assertTrue(control.upper.at_home())
        self.assertTrue(control.active)  # overlay can wait for readback without blocking walking
        self.assertEqual(tick(request=(0.12, 0.0, 0.0)), [0.12, 0.0, 0.0])
        self.geometry()  # measured joints reach the returned Home targets
        for _ in range(50):
            tick()
        self.assertFalse(control.active)
        self.assertEqual(tick(request=(0.12, 0.0, 0.0)), [0.12, 0.0, 0.0])

    def test_walk_turn_and_reverse_hold_nonhome_arm_targets(self):
        self.geometry()
        control = ArmTeleop(self.model, self.data, self.home)
        for _ in range(35):
            control.gate([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.02)
            control.apply(self.targets, self.names, {**self.state, "ry": -1.0, "a": True}, 0.02)
        self.assertTrue(control.active)
        self.assertFalse(control.upper.at_home())
        held = dict(control.upper.positions)
        for requested in ([0.12, 0.0, 0.0], [-0.06, 0.0, 0.0], [0.0, 0.0, 0.25], [0.08, 0.0, -0.2]):
            self.assertEqual(control.gate(requested, requested, 0.02), requested)
            self.targets[:] += 0.01  # simulate fresh policy targets each tick
            control.apply(
                self.targets, self.names, {**self.state, "rx": 1.0, "y_button": True}, 0.02
            )
            self.assertEqual(control.upper.positions, held)
            for n, value in held.items():
                self.assertAlmostEqual(self.targets[self.names.index(n)], value)
        control.reset()
        self.assertFalse(control.active)
        self.assertFalse(control.ready)
        self.assertTrue(control.upper.at_home())
        self.assertFalse(control.upper.returning)
        self.assertIsNone(control.previous)

    def test_updated_motion_times(self):
        self.step(60, a=True)
        self.assertAlmostEqual(self.upper.positions[FINGER_NAMES[0]], self.upper.open_position)
        self.step(57, y_button=True)  # full 27 mm closing travel in <= 1.14 s
        self.assertAlmostEqual(self.upper.positions[FINGER_NAMES[0]], self.upper.closed_position)
        self.step(55, ry=-1.0)  # full 0.70 rad lift in <= 1.10 s
        self.assertAlmostEqual(self.upper.lift, self.upper.lift_max)

    def test_walking_preserves_policy_arm_targets(self):
        self.geometry()
        control = ArmTeleop(self.model, self.data, self.home)
        for n in ARM_NAMES:
            self.targets[self.names.index(n)] += 0.01
        before = self.targets.copy()
        self.assertEqual(control.gate([0.1, 0.0, 0.1], [0.0, 0.0, 0.0], 0.02), [0.1, 0.0, 0.1])
        control.apply(self.targets, self.names, {**self.state, "rx": 1.0, "a": True}, 0.02)
        np.testing.assert_array_equal(before, self.targets)

    def test_actuated_upper_body_tracks_targets_in_physics(self):
        # A supported torso isolates actuator direction/tracking from balance.
        self.step(70, rx=1.0, ry=-1.0, a=True)
        goal = self.targets.copy()
        for n, v in self.home.items():
            self.data.qpos[self.model.joint(n).qposadr[0]] = v
        self.model.opt.gravity[:] = 0.0
        try:
            for _ in range(1000):
                for i in range(self.model.nu):
                    name = self.model.actuator(i).name
                    self.data.ctrl[i] = goal[self.names.index(name)]
                mujoco.mj_step(self.model, self.data)
                # Hold floating base at its initial pose for this bench check.
                self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
                self.data.qvel[:6] = 0.0
            for n in ARM_NAMES + FINGER_NAMES:
                tolerance = 0.002 if n in FINGER_NAMES else 0.03
                self.assertAlmostEqual(
                    self.data.qpos[self.model.joint(n).qposadr[0]],
                    goal[self.names.index(n)],
                    delta=tolerance,
                )
        finally:
            self.model.opt.gravity[:] = [0.0, 0.0, -9.81]


if __name__ == "__main__":
    unittest.main()
