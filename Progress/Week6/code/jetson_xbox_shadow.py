"""Xbox input for the Jetson R25 policy, limited to read-only shadow runs.

This file deliberately has no motor-output mode. It reads a Linux evdev
controller, applies the Week 5 left-stick mapping, and feeds commands to the
existing R25 shadow runner. The runner reads robot sensors and computes policy
targets but never enables torque or sends motor goals.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import struct
import sys
import time


ABS_X = 0
ABS_Y = 1
BTN_EAST = 305  # B on a standard Xbox mapping
BTN_TL = 310  # LB on a standard Xbox mapping
KEY_BYTES = 96
ABS_INFO = struct.Struct("6i")
DEAD_ZONE = 0.24
VECTOR_LENGTH = 0.30
VECTOR_DEAD_ZONE = 0.015
LIMITS = {"forward": 0.06, "reverse": 0.04, "yaw": 0.15}
ZERO = (0.0, 0.0, 0.0)


def _read_ioctl(number: int, size: int) -> int:
    return (2 << 30) | (size << 16) | (ord("E") << 8) | number


def axis_info(fd: int, code: int) -> tuple[int, int, int]:
    data = bytearray(ABS_INFO.size)
    fcntl.ioctl(fd, _read_ioctl(0x40 + code, len(data)), data, True)
    value, low, high, *_ = ABS_INFO.unpack(data)
    if low >= high:
        raise ValueError(f"ABS axis {code} has invalid limits {low}..{high}")
    return value, low, high


def button_down(fd: int, code: int) -> bool:
    data = bytearray(KEY_BYTES)
    fcntl.ioctl(fd, _read_ioctl(0x18, len(data)), data, True)
    if code < 0 or code // 8 >= len(data):
        raise ValueError(f"Button code {code} is out of range")
    return bool(data[code // 8] & (1 << (code % 8)))


def normalize(value: int, low: int, high: int) -> float:
    center = (low + high) / 2
    distance = (high - center) if value >= center else (center - low)
    return max(-1.0, min(1.0, (value - center) / distance))


def dead_zone(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Nonfinite controller axis")
    value = max(-1.0, min(1.0, value))
    return math.copysign(max(0.0, (abs(value) - DEAD_ZONE) / (1 - DEAD_ZONE)), value)


def map_left_stick(x: float, y: float, limits: dict[str, float] = LIMITS) -> tuple[float, float, float]:
    """Match Week 5's body-frame mapping within the tested R25 bench limits."""
    forward = -dead_zone(y) * VECTOR_LENGTH
    turn = -dead_zone(x) * VECTOR_LENGTH
    length = math.hypot(forward, turn)
    if length > VECTOR_LENGTH:
        forward *= VECTOR_LENGTH / length
        turn *= VECTOR_LENGTH / length

    def scale(value: float, limit: float, minimum: float) -> float:
        if abs(value) <= VECTOR_DEAD_ZONE or limit <= 0:
            return 0.0
        amount = min(1.0, (abs(value) - VECTOR_DEAD_ZONE) / (VECTOR_LENGTH - VECTOR_DEAD_ZONE))
        return math.copysign(min(minimum, limit) + (limit - min(minimum, limit)) * amount, value)

    return (
        scale(forward, limits["forward"] if forward >= 0 else limits["reverse"],
              0.03 if forward >= 0 else 0.04),
        0.0,
        scale(turn, limits["yaw"], 0.08),
    )


class Deadman:
    """Require neutral stick and released LB before every arming cycle."""

    def __init__(self) -> None:
        self.armed = False
        self.reason = "waiting for controller"

    def update(self, *, connected: bool, x: float = 0.0, y: float = 0.0,
               lb: bool = False, b: bool = False) -> tuple[float, float, float]:
        if not connected:
            self.armed = False
            self.reason = "controller disconnected"
            return ZERO
        if b:
            self.armed = False
            self.reason = "B stop; release LB and centre stick to re-arm"
            return ZERO
        neutral = dead_zone(x) == 0.0 and dead_zone(y) == 0.0
        if neutral and not lb:
            self.armed = True
            self.reason = "ready; hold LB to command motion"
            return ZERO
        if not lb:
            self.armed = False
            self.reason = "LB released; centre stick to re-arm"
            return ZERO
        if not self.armed:
            self.reason = "release LB and centre stick to re-arm"
            return ZERO
        command = map_left_stick(x, y)
        self.reason = "commanded" if command != ZERO else "stick centred"
        return command


def controller_candidates() -> list[tuple[Path, str]]:
    result = []
    for sys_path in sorted(Path("/sys/class/input").glob("event*")):
        try:
            name = (sys_path / "device/name").read_text().strip()
        except OSError:
            continue
        if any(word in name.lower() for word in ("xbox", "x-box", "microsoft", "gamepad")):
            result.append((Path("/dev/input") / sys_path.name, name))
    return result


class Gamepad:
    def __init__(self, device: Path | None = None, x_code: int = ABS_X,
                 y_code: int = ABS_Y, lb_code: int = BTN_TL,
                 stop_code: int = BTN_EAST) -> None:
        self.requested_device = device
        self.x_code, self.y_code = x_code, y_code
        self.lb_code, self.stop_code = lb_code, stop_code
        self.path: Path | None = None
        self.fd: int | None = None
        self.name = ""
        self.deadman = Deadman()

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.fd = None
        self.path = None
        self.name = ""
        self.deadman.update(connected=False)

    def _open(self) -> None:
        choices = ([(self.requested_device, self.requested_device.name)]
                   if self.requested_device else controller_candidates())
        for path, name in choices:
            if path is None or not path.exists():
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            except PermissionError as exc:
                raise PermissionError(f"Cannot read {path}; grant access to this Xbox event device only") from exc
            try:
                axis_info(fd, self.x_code)
                axis_info(fd, self.y_code)
            except (OSError, ValueError):
                os.close(fd)
                continue
            self.fd, self.path, self.name = fd, path, name
            print(f"XBOX_CONNECTED {name} {path}", flush=True)
            return

    def sample(self) -> dict[str, object]:
        if self.fd is None:
            self._open()
        if self.fd is not None:
            try:
                if not self.path or not self.path.exists() or os.stat(self.path).st_rdev != os.fstat(self.fd).st_rdev:
                    raise OSError("Controller device disappeared")
                x = normalize(*axis_info(self.fd, self.x_code))
                y = normalize(*axis_info(self.fd, self.y_code))
                lb = button_down(self.fd, self.lb_code)
                b = button_down(self.fd, self.stop_code)
            except OSError:
                print("XBOX_DISCONNECTED", flush=True)
                self.close()
            else:
                command = self.deadman.update(connected=True, x=x, y=y, lb=lb, b=b)
                return {"connected": True, "device": str(self.path), "name": self.name,
                        "x": round(x, 4), "y": round(y, 4), "lb": lb, "b": b,
                        "armed": self.deadman.armed, "reason": self.deadman.reason,
                        "command": command}
        self.deadman.update(connected=False)
        return {"connected": False, "device": None, "name": "", "x": 0.0, "y": 0.0,
                "lb": False, "b": False, "armed": False,
                "reason": self.deadman.reason, "command": ZERO}


def scan() -> None:
    candidates = controller_candidates()
    if not candidates:
        print("NO_XBOX_INPUT_DEVICE")
    for path, name in candidates:
        print(json.dumps({"path": str(path), "name": name,
                          "readable": os.access(path, os.R_OK)}))


def monitor(pad: Gamepad, seconds: float) -> None:
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            print(json.dumps(pad.sample()), flush=True)
            time.sleep(0.1)
    finally:
        pad.close()


def shadow(pad: Gamepad, args: argparse.Namespace) -> None:
    """Inject live commands only into the existing no-motor R25 shadow path."""
    first = pad.sample()
    if not first["connected"]:
        raise RuntimeError("No readable Xbox controller; run scan/monitor after connecting it")
    import r25_rollout

    original_history = r25_rollout.R25History
    original_argv = sys.argv
    log_path = Path(__file__).resolve().parent / f"xbox_input_{time.time_ns()}.jsonl"
    with log_path.open("x", encoding="utf-8") as input_log:
        class XboxHistory(original_history):
            def advance_command(self, _requested: object) -> None:
                state = pad.sample()
                input_log.write(json.dumps({"time": time.time(), **state}) + "\n")
                input_log.flush()
                super().advance_command(state["command"])

        try:
            r25_rollout.R25History = XboxHistory
            sys.argv = ["r25_rollout.py", "shadow", "--seconds", str(args.seconds),
                        "--delay", str(args.delay), "--command", "stop"]
            if args.allow_uncalibrated_accel:
                sys.argv.append("--allow-uncalibrated-accel")
            if args.fixed_body_vx is not None:
                sys.argv.extend(("--experimental-fixed-body-vx", str(args.fixed_body_vx)))
            r25_rollout.main()
        finally:
            r25_rollout.R25History = original_history
            sys.argv = original_argv
            pad.close()
    print(f"XBOX_INPUT_LOG {log_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "monitor", "shadow"))
    parser.add_argument("--device", type=Path)
    parser.add_argument("--x-code", type=int, default=ABS_X)
    parser.add_argument("--y-code", type=int, default=ABS_Y)
    parser.add_argument("--lb-code", type=int, default=BTN_TL)
    parser.add_argument("--stop-code", type=int, default=BTN_EAST)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--allow-uncalibrated-accel", action="store_true")
    parser.add_argument("--fixed-body-vx", type=float)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.delay <= 60:
        parser.error("seconds must be positive and delay must be 0..60")
    if args.fixed_body_vx is not None and (not math.isfinite(args.fixed_body_vx) or
                                           not 0 <= args.fixed_body_vx <= 0.12):
        parser.error("fixed body vx must be 0..0.12 m/s")
    if args.mode == "scan":
        scan()
        return
    pad = Gamepad(args.device, args.x_code, args.y_code, args.lb_code, args.stop_code)
    if args.mode == "monitor":
        monitor(pad, args.seconds)
    else:
        shadow(pad, args)


if __name__ == "__main__":
    main()
