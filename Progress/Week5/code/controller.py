"""Bounded gamepad input, independent of the simulation runtime."""
import math
import threading
import time

DEAD_ZONE = 0.24
TIMEOUT = 0.30
LIMITS = dict(forward_limit=0.12, backward_limit=0.06, yaw_limit=0.25)


def axis(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Invalid axis')
    value = max(-1., min(1., value))
    return math.copysign(max(0., (abs(value)-DEAD_ZONE)/(1.-DEAD_ZONE)), value)


def map_command(x, y, limits):
    """Left-stick Y is throttle; left-stick X steers. No strafe."""
    throttle, steering = -axis(y), -axis(x)
    def scaled(v, limit, minimum):
        return math.copysign(min(minimum, limit)+(limit-min(minimum, limit))*abs(v), v) if v else 0.
    return [scaled(throttle, limits['forward_limit'] if throttle >= 0 else limits['backward_limit'],
                   .03 if throttle >= 0 else .04), 0., scaled(steering, limits['yaw_limit'], .08)]


class Controller:
    def __init__(self, clock=time.monotonic, limits=None, require_lb=True, mapper=map_command):
        self.clock = clock
        self.limits = dict(limits or LIMITS)
        self.require_lb = require_lb
        self.mapper = mapper
        self.lock = threading.RLock()
        self.owner = None
        self.last_time = -math.inf
        self.last_seq = -1
        self.armed = False
        self.reason = 'Waiting for controller'
        self.device = None
        self.packet = None
        self.received = self.moving = 0
        self.reset_pending = False
        self.previous_menu = False

    def stop(self, reason='Stopped; release LB and centre stick to re-arm'):
        with self.lock:
            self.armed = False
            self.reason = reason
            self.packet = None

    def submit(self, p):
        # Validate the entire packet before updating its timestamp.
        if not isinstance(p, dict):
            raise ValueError('Input must be an object')
        client, seq = p.get('client'), p.get('seq')
        if not isinstance(client, str) or not 1 <= len(client) <= 80 or type(seq) is not int or seq < 0:
            raise ValueError('Invalid client/sequence')
        x, y = p.get('x', 0.), p.get('y', 0.)
        axis(x); axis(y)
        for key in ('connected', 'focused', 'lb', 'b', 'menu'):
            if type(p.get(key)) is not bool:
                raise ValueError('Invalid button/state')
        if not isinstance(p.get('id'), str) or len(p['id']) > 256:
            raise ValueError('Invalid device')
        if not isinstance(p.get('mapping'), str):
            raise ValueError('Invalid mapping')
        with self.lock:
            now = self.clock()
            expired = now-self.last_time > TIMEOUT
            if self.owner != client:
                if self.owner and not expired:
                    return False
                self.owner, self.last_seq = client, -1
                self.stop('New browser; release LB and centre stick')
                self.device = None
            if seq <= self.last_seq:
                return False
            if expired:
                self.stop('Input timed out; release LB and centre stick')
                self.previous_menu = False
            if self.device != p['id']:
                self.stop('Controller changed; release LB and centre stick')
                self.previous_menu = False
            self.last_seq, self.last_time = seq, now
            self.packet, self.device = dict(p), p['id']
            self.received += 1
            neutral = axis(x) == 0. and axis(y) == 0.
            if not p['connected']:
                self.armed, self.reason = False, 'Controller disconnected / press A to detect'
            elif p['mapping'] != 'standard':
                self.armed, self.reason = False, 'Unsupported gamepad mapping'
            elif not p['focused']:
                self.armed, self.reason = False, 'Window inactive'
            elif p['b']:
                self.armed, self.reason = False, 'B stop; release LB and centre stick'
            elif neutral and not p['lb']:
                self.armed, self.reason = True, ('Ready; hold LB to move' if self.require_lb else 'Ready; move sticks to walk/turn')
            elif not self.armed:
                self.reason = 'Release LB and centre stick to re-arm'
            elif self.require_lb and not p['lb']:
                self.reason = 'LB released; zero command'
            else:
                self.reason = 'Driving' if not neutral else 'Ready; stick centred'
            if self.armed and p['connected'] and p['focused'] and p['mapping'] == 'standard' and not p['b'] and (p['lb'] or not self.require_lb) and any(self.mapper(x, y, self.limits)):
                self.moving += 1
            if p['menu'] and not self.previous_menu and not p['lb'] and self.armed:
                self.reset_pending = True
                self.armed = False
                self.reason = 'Reset requested'
            self.previous_menu = p['menu']
            return True

    def sample(self):
        with self.lock:
            age = self.clock()-self.last_time
            if age > TIMEOUT:
                self.stop('Input timeout; release LB and centre stick')
            p = self.packet
            enabled = bool(self.armed and p and p['connected'] and p['focused'] and
                           p['mapping'] == 'standard' and (p['lb'] or not self.require_lb) and not p['b'])
            command = self.mapper(p['x'], p['y'], self.limits) if enabled else [0., 0., 0.]
            return dict(command=command, armed=self.armed, reason=self.reason,
                        device=self.device, connected=bool(p and p['connected']),
                        mapping=p['mapping'] if p else None,
                        x=p['x'] if p else 0., y=p['y'] if p else 0.,
                        lb=bool(p and p['lb']), age_s=age if math.isfinite(age) else None,
                        packets=self.received, nonzero_input_packets=self.moving, require_lb=self.require_lb,
                        limits=self.limits)

    def take_reset(self):
        with self.lock:
            requested, self.reset_pending = self.reset_pending, False
            return requested
