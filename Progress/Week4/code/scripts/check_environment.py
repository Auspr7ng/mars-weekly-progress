"""Check the packaged inference contract without starting training or a server."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import sys


def main():
    week=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(week/'code/vendor'))
    sys.path.insert(0,str(week/'code/src'))
    if sys.version_info[:2]!=(3,11):raise RuntimeError('The validated environment uses Python 3.11')
    expected={'mujoco':'3.8.1','mujoco-warp':'3.8.1','warp-lang':'1.13.0','torch':'2.7.1','rsl-rl-lib':'5.2.0'}
    versions={name:metadata.version(name) for name in expected}
    for name,version in expected.items():
        if versions[name].split('+')[0]!=version:raise RuntimeError(f'{name}: expected {version}, found {versions[name]}')
    import viser
    if not hasattr(viser.IcosphereHandle,'on_drag'):raise RuntimeError('The viewer needs the included Viser drag-lifecycle API')
    if not (Path(viser.__file__).parent/'client/build/index.html').is_file():raise RuntimeError('The prebuilt Viser client is missing')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for mjlab initialization and distance sensors')
    manifest=json.loads((week/'config/accepted_policy.json').read_text(encoding='utf-8'))
    checkpoint=week/manifest['checkpoint'];policy=checkpoint.parent
    for path,key in [(checkpoint,'checkpoint_sha256'),(policy/'dynamics/sunday_a1_locomotion.xml','model_sha256'),(policy/'dynamics/arm_home.json','nominal_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[key]:raise RuntimeError(f'Accepted artifact hash mismatch: {path.name}')
    run=json.loads((policy/'run.json').read_text(encoding='utf-8'))
    for name,expected_hash in run['source_hashes'].items():
        if hashlib.sha256((policy/'source'/name).read_bytes()).hexdigest()!=expected_hash:raise RuntimeError(f'Frozen source hash mismatch: {name}')
    print(json.dumps(dict(passed=True,python=sys.version.split()[0],dependencies=versions,
        cuda_available=True,viser_source=str(Path(viser.__file__).relative_to(week)),
        policy_hash_verified=True,frozen_source_hashes_verified=True),indent=2))


if __name__=='__main__':main()
