"""Open the accepted local walking demo, reusing a running matching server."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import webbrowser


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8086);p.add_argument('--check',action='store_true');args=p.parse_args()
    week=Path(__file__).resolve().parents[1]
    if args.check:
        import runpy
        runpy.run_path(str(week/'code/scripts/check_environment.py'),run_name='__main__')
        return
    manifest=json.loads((week/'config/accepted_policy.json').read_text(encoding='utf-8'))
    checkpoint=week/manifest['checkpoint']
    if not manifest.get('accepted') or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise RuntimeError('Accepted policy manifest/checkpoint mismatch')
    url=f'http://127.0.0.1:{args.port}'
    with socket.socket() as connection:
        connection.settimeout(.3);running=connection.connect_ex(('127.0.0.1',args.port))==0
    if running:
        info_path=week/'results/web_demo/server_info.json'
        info=json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
        if info.get('checkpoint_sha256')!=manifest['checkpoint_sha256'] or info.get('port')!=args.port:
            raise RuntimeError(f'Port {args.port} is already in use; choose another --port or close the previous demo')
        if info.get('interaction_mode')!=manifest.get('interaction_mode') or info.get('joystick_limits')!=manifest.get('joystick_limits') or info.get('mapper_version')!='stable_floor_v1' or info.get('interaction_ui_version')!='body_follow_v3' or info.get('physics_backend')!='native_mujoco' or info.get('actor_device')!='cpu':
            raise RuntimeError('The running demo uses an earlier control profile. Close its viewer, then start this launcher again.')
        webbrowser.open(url);print('Robot demo is running: '+url,flush=True);return
    raise SystemExit(subprocess.call([sys.executable,'-B','-u',str(week/'code/scripts/play.py'),
                       '--accepted-policy','--arrow-control','--port',str(args.port),'--open-browser'],cwd=str(week)))


if __name__=='__main__':main()
