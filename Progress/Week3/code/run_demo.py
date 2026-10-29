"""Relocatable Week3 entry point; reuse an existing mjlab Python environment."""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import os
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'code'
# Load this week's task package and reuse the installed mjlab environment.
sys.path.insert(0, str(DEMO/'src'))
os.environ.setdefault('MPLCONFIGDIR', str(DEMO/'.cache/matplotlib'))


def prepare():
    required = {'mjlab': '1.4.0', 'mujoco': '3.8.1', 'mujoco-warp': '3.8.1', 'warp-lang': '1.13.0', 'torch': '2.7.1'}
    problems = []
    for name, expected in required.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f'{name}: missing (expected {expected})')
            continue
        if actual.split('+')[0] != expected:
            problems.append(f'{name}: {actual} (expected {expected})')
    if problems:
        raise RuntimeError('Incompatible runtime dependencies:\n  '+'\n  '.join(problems)+
                           '\nSelect a compatible mjlab Python with code/configure_python.bat. See README.md.')
    try:
        import torch
        import warp as wp
        import mujoco
        import PIL
        import matplotlib
        import viser
        import mjlab
        import mjlab.tasks
        import sunday_a1_mjlab.tasks
    except ImportError as exc:
        raise RuntimeError('Missing dependency in selected Python. Use the Python from your mjlab environment: '+str(exc)) from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select the CUDA-enabled mjlab environment and check the NVIDIA driver.')
    wp.config.kernel_cache_dir = str(DEMO/'.cache/warp')
    from sunday_a1_mjlab import tasks
    if not Path(tasks.__file__).resolve().is_relative_to(DEMO/'src'):
        raise RuntimeError('Sunday A1 task was imported from outside this bundle')
    print('Week3 project:', DEMO, flush=True)
    print('mjlab source:', mjlab.__file__, flush=True)
    return torch, mujoco


def importlib_module_path(name):
    return Path(sys.modules[name].__file__).resolve()


def check():
    torch, mujoco = prepare()
    from sunday_a1_mjlab.ball_grasp import BallGraspBackend, GRIP_TARGET
    source_urdf = DEMO/'model/sunday_a1.urdf'
    if not source_urdf.is_file():
        raise RuntimeError('Bundled model is missing: '+str(source_urdf))
    backend = BallGraspBackend()
    runtime = backend.runtime
    try:
        for _ in range(30):
            hand = runtime.step([GRIP_TARGET]*4)
        foot = runtime.env.scene['feet_tactile'].data
        assert tuple(hand.response.shape) == (1,4,384)
        assert tuple(foot.response.shape) == (1,4,1024)
        assert torch.isfinite(hand.response).all() and torch.isfinite(foot.response).all()
        assert runtime.model.nu == 27
        report = {'passed': True, 'python': sys.executable, 'bundle_root': str(ROOT),
                  'project_source': str(DEMO/'src'), 'mjlab_source': str(importlib_module_path('mjlab')),
                  'gpu': torch.cuda.get_device_name(), 'hand_shape': list(hand.response.shape),
                  'foot_shape': list(foot.response.shape), 'actuators': runtime.model.nu,
                  'source_model': 'code/model/sunday_a1.urdf', 'steps':30,
                  'runtime_versions': {n:metadata.version(n) for n in ['mujoco','mujoco-warp','warp-lang','torch','viser']}}
        dest=ROOT/'results/validation'
        dest.mkdir(exist_ok=True)
        (dest/'bundle_runtime_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        backend.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--check',action='store_true')
    selection.add_argument('--regressions',action='store_true')
    selection.add_argument('--script',help='A script filename inside code/scripts')
    args,extra=parser.parse_known_args()
    os.chdir(DEMO)
    if args.check:
        check()
        return
    if args.regressions:
        for filename in ('check_dual_ball_grasp.py','check_feet_tactile.py','check_flexitac_mjlab.py','check_tactile_objects.py'):
            subprocess.run([sys.executable,str(DEMO/'run_demo.py'),'--script',filename],check=True)
        return
    filename=args.script or 'ball_grasp_demo.py'
    script=(DEMO/'scripts'/filename).resolve()
    if script.parent != (DEMO/'scripts').resolve() or not script.is_file() or script.suffix != '.py':
        parser.error('Choose an existing Python filename in code/scripts')
    # Conversion/static preview also work without loading a GPU environment.
    if filename not in ('convert_robot.py','preview_robot.py','render_preview.py'):
        prepare()
    sys.argv=[str(script),*extra]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,ImportError) as exc:
        print('\nWeek3 startup failed: '+str(exc),file=sys.stderr)
        sys.exit(1)
