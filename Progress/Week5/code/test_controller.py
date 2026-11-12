"""Input contract tests; these do not certify real controller/model operation."""
import math
import unittest
from controller import Controller, LIMITS, map_command


class InputContract(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.c = Controller(clock=lambda: self.now)
        self.seq = 0

    def send(self, **kw):
        self.seq += 1
        p = dict(client='test', seq=self.seq, id='Xbox test', mapping='standard',
                 connected=True, focused=True, x=0., y=0., lb=False, b=False, menu=False)
        p.update(kw)
        return self.c.submit(p)

    def test_limits_and_signs(self):
        self.assertEqual(map_command(-1, -1, LIMITS), [.12, 0., .25])
        self.assertEqual(map_command(1, 1, LIMITS), [-.06, 0., -.25])

    def test_direct_control_needs_no_lb(self):
        self.c = Controller(clock=lambda: self.now, require_lb=False)
        self.send(y=-1.)
        self.assertEqual(self.c.sample()['command'], [0.,0.,0.])
        self.send()
        self.send(y=-1.)
        self.assertEqual(self.c.sample()['command'], [.12,0.,0.])
        self.send()
        self.assertEqual(self.c.sample()['command'], [0.,0.,0.])

    def test_direct_stop_does_not_resume_deflected_stick(self):
        self.c = Controller(clock=lambda: self.now, require_lb=False)
        self.send(); self.send(y=-1.); self.send(y=-1., b=True); self.send(y=-1.)
        self.assertEqual(self.c.sample()['command'], [0.,0.,0.])

    def test_deadzone(self):
        self.assertEqual(map_command(.24, -.24, LIMITS), [0., 0., 0.])

    def test_startup_requires_neutral_and_lb_release(self):
        self.send(lb=True, y=-1.)
        self.assertEqual(self.c.sample()['command'], [0., 0., 0.])
        self.send()
        self.send(lb=True, y=-1.)
        self.assertEqual(self.c.sample()['command'][0], .12)

    def test_lb_release_stops(self):
        self.send(); self.send(lb=True, y=-1.); self.send(lb=False, y=-1.)
        self.assertEqual(self.c.sample()['command'], [0., 0., 0.])

    def test_disconnect_and_reconnect(self):
        self.send(); self.send(lb=True, y=-1.); self.send(connected=False)
        self.send(lb=True, y=-1.)
        self.assertEqual(self.c.sample()['command'], [0., 0., 0.])

    def test_stale_input_requires_rearm(self):
        self.send(); self.send(lb=True, y=-1.)
        self.now = .31
        self.assertEqual(self.c.sample()['command'], [0., 0., 0.])
        self.send(lb=True, y=-1.)
        self.assertFalse(self.c.sample()['armed'])

    def test_focus_loss(self):
        self.send(); self.send(lb=True, y=-1.); self.send(focused=False, lb=True, y=-1.)
        self.assertEqual(self.c.sample()['command'], [0., 0., 0.])

    def test_stop_requires_rearm(self):
        self.send(); self.send(lb=True, y=-1.)
        self.c.stop(); self.send(lb=True, y=-1.)
        self.assertFalse(self.c.sample()['armed'])

    def test_b_and_unknown_mapping(self):
        self.send(); self.send(lb=True, y=-1., b=True)
        self.assertEqual(self.c.sample()['command'], [0., 0., 0.])
        self.send(mapping='')
        self.assertFalse(self.c.sample()['armed'])

    def test_replay_or_other_client_rejected(self):
        self.send()
        self.assertFalse(self.send(seq=0))
        self.assertFalse(self.send(client='other', lb=True, y=-1.))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.send(x=math.nan)

    def test_reset_only_with_lb_released(self):
        self.send(); self.send(lb=True, menu=True)
        self.assertFalse(self.c.take_reset())
        self.send(); self.send(menu=True)
        self.assertTrue(self.c.take_reset())
        self.assertFalse(self.c.take_reset())


if __name__ == '__main__':
    unittest.main()
