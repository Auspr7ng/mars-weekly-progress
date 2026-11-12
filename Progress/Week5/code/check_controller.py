"""Read-only Xbox diagnostics. Never loads a model or sends motor commands."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

WEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK / '.deps'))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


class Gamepad(ctypes.Structure):
    _fields_ = [('buttons', wintypes.WORD), ('lt', wintypes.BYTE), ('rt', wintypes.BYTE),
                ('lx', ctypes.c_short), ('ly', ctypes.c_short),
                ('rx', ctypes.c_short), ('ry', ctypes.c_short)]


class State(ctypes.Structure):
    _fields_ = [('packet', wintypes.DWORD), ('gamepad', Gamepad)]


def xinput_probe():
    ctypes.OleDLL('ole32').CoInitialize(None)
    reports = []
    for name in ('xinput1_4.dll', 'xinput1_3.dll', 'xinput9_1_0.dll'):
        try:
            dll = ctypes.WinDLL(name)
            dll.XInputGetState.argtypes = [wintypes.DWORD, ctypes.POINTER(State)]
            dll.XInputGetState.restype = wintypes.DWORD
            states = []
            for i in range(4):
                s = State()
                result = dll.XInputGetState(i, ctypes.byref(s))
                states.append(dict(slot=i, result_hex=hex(result), connected=result == 0,
                                   axes=[s.gamepad.lx, s.gamepad.ly, s.gamepad.rx, s.gamepad.ry],
                                   buttons=s.gamepad.buttons))
            reports.append(dict(dll=name, slots=states))
        except OSError as e:
            reports.append(dict(dll=name, error=str(e)))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15.)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error('--seconds must be in (0, 120]')
    report = dict(timestamp=datetime.now().astimezone().isoformat(),
                  purpose='Read-only controller diagnostics; no model or motor control',
                  xinput=xinput_probe(), devices=[], samples=[])
    print('XInput:', json.dumps(report['xinput']), flush=True)
    try:
        import pygame
        from pygame._sdl2 import controller as sdl_controller
    except ImportError:
        report['pygame_error'] = 'pygame not installed in Week5/.deps'
        print(report['pygame_error'], flush=True)
    else:
        # SDL event processing must run on the thread that initializes video.
        pygame.display.init()
        pygame.display.set_mode((520, 100))
        pygame.display.set_caption('Week5 Xbox input test - move sticks / close to exit')
        pygame.joystick.init()
        sdl_controller.init()
        opened = {}
        last_print = 0.
        started = time.monotonic()
        try:
            while time.monotonic()-started < args.seconds:
                if any(e.type == pygame.QUIT for e in pygame.event.get()):
                    break
                for index in range(pygame.joystick.get_count()):
                    joy = pygame.joystick.Joystick(index)
                    instance = joy.get_instance_id()
                    if instance not in opened:
                        mapped = sdl_controller.Controller(index) if sdl_controller.is_controller(index) else None
                        opened[instance] = (joy, mapped)
                        info = dict(name=joy.get_name(), guid=joy.get_guid(), instance=instance,
                                    axes=joy.get_numaxes(), buttons=joy.get_numbuttons(),
                                    standard_mapping=mapped is not None)
                        report['devices'].append(info)
                        print('DEVICE:', json.dumps(info), flush=True)
                if time.monotonic()-last_print >= .2:
                    last_print = time.monotonic()
                    for instance, (joy, mapped) in list(opened.items()):
                        try:
                            sample = dict(t=round(last_print-started, 3), instance=instance,
                                          raw_axes=[round(joy.get_axis(i), 4) for i in range(joy.get_numaxes())],
                                          raw_buttons=[joy.get_button(i) for i in range(joy.get_numbuttons())])
                            if mapped is not None:
                                sample['standard_axes'] = dict(left_x=mapped.get_axis(pygame.CONTROLLER_AXIS_LEFTX), left_y=mapped.get_axis(pygame.CONTROLLER_AXIS_LEFTY),
                                                               right_x=mapped.get_axis(pygame.CONTROLLER_AXIS_RIGHTX))
                                sample['standard_buttons'] = dict(lb=mapped.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER),
                                                                  b=mapped.get_button(pygame.CONTROLLER_BUTTON_B))
                            report['samples'].append(sample)
                            print(json.dumps(sample), flush=True)
                        except pygame.error:
                            del opened[instance]
                time.sleep(.01)
        finally:
            pygame.quit()
    report['detected'] = bool(report['devices']) or any(s['connected'] for d in report['xinput'] for s in d.get('slots', []))
    report['physical_input_verified'] = any(any(abs(v) > 8000 for v in s.get('standard_axes', {}).values()) or any(s['raw_buttons'])
                                            for s in report['samples'])
    out = WEEK / 'results/controller_diagnostics.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Detected:', report['detected'], '| Input observed:', report['physical_input_verified'], flush=True)
    print('Report:', out, flush=True)


if __name__ == '__main__':
    main()
