"""SDL standard controller input. SDL owns its event loop in one worker thread."""
import json
import os
from pathlib import Path
import sys
import threading
import time

WEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK / '.deps'))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


class NativeInput:
    def __init__(self, control):
        self.control = control
        self.on_reset = None
        self.halt = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name='Xbox-SDL', daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(8):
            raise RuntimeError('SDL input initialization timed out')
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.halt.set()
        self.thread.join(timeout=3)
        self.control.stop('Viewer closed')

    def run(self):
        try:
            import pygame
            from pygame._sdl2 import controller as sdl
            pygame.display.init()
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            pygame.joystick.init()
            sdl.init()
            pad = None
            seq = 0
            last_log = 0.
            self.ready.set()
            with (WEEK / 'results/input_trace.jsonl').open('a', encoding='utf-8', buffering=1) as log:
                while not self.halt.is_set():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.halt.set()
                    if pad is not None and not pad.attached():
                        pad.quit()
                        pad = None
                        self.control.stop('Controller disconnected')
                    if pad is None:
                        for i in range(pygame.joystick.get_count()):
                            if sdl.is_controller(i):
                                pad = sdl.Controller(i)
                                self.control.stop('Controller connected; release LB and centre sticks')
                                print('XBOX_CONNECTED ' + pad.name, flush=True)
                                break
                    seq += 1
                    def normalized(value):
                        return value/(32768. if value < 0 else 32767.)
                    try:
                        p = dict(client='native-sdl', seq=seq, id=pad.name if pad else '',
                                 mapping='standard' if pad else '', connected=pad is not None,
                                 focused=True,
                                 x=normalized(pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX)) if pad else 0.,
                                 y=normalized(pad.get_axis(pygame.CONTROLLER_AXIS_LEFTY)) if pad else 0.,
                                 lb=bool(pad and pad.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER)),
                                 b=bool(pad and pad.get_button(pygame.CONTROLLER_BUTTON_B)),
                                 menu=bool(pad and pad.get_button(pygame.CONTROLLER_BUTTON_START)))
                    except pygame.error:
                        # A Bluetooth disconnect can race the attached() check.
                        # Drop the old command, then enumerate again on the next tick.
                        self.control.stop('Controller disconnected during read')
                        try:
                            if pad:
                                pad.quit()
                        except pygame.error:
                            pass
                        pad = None
                        self.halt.wait(.02)
                        continue
                    self.control.submit(p)
                    if self.control.take_reset() and self.on_reset:
                        self.on_reset()
                    now = time.monotonic()
                    if now-last_log >= .1:
                        last_log = now
                        log.write(json.dumps(dict(time=time.time(), input=p, state=self.control.sample()))+'\n')
                    self.halt.wait(.02)
        except Exception as e:
            self.error = repr(e)
            self.control.stop('Input error: '+str(e))
            self.ready.set()
            print('XBOX_INPUT_ERROR '+self.error, flush=True)
        finally:
            try:
                pygame.quit()
            except (NameError, UnboundLocalError):
                pass
            self.control.stop('Input stopped')
