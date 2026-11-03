"""Verify unchanged Week3 distance sensing on the current floating model."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import hashlib
import json
import mujoco
import numpy as np
from week3_tactile.flexitac import FourFingerFlexiTac
from week3_tactile.feet_tactile import FootTactileEngine
from week3_tactile.tactile_display import save_foot_snapshot,save_snapshot


def main():
    home=json.loads((WEEK/'model/home.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'))
    data=mujoco.MjData(model)
    data.qpos[2]=home['root_height']
    for name,value in home['joint_pos'].items():data.qpos[model.joint(name).qposadr[0]]=value
    data.ctrl[:]=[home['joint_pos'][model.actuator(i).name] for i in range(model.nu)]
    for _ in range(1500):mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    feet=FootTactileEngine(model)
    hands=FourFingerFlexiTac(model)
    result=feet.sample_mujoco(data)
    foot_response=result.response.detach().cpu().numpy().copy()
    points=result.tactile_points_w_per_sensor.detach().cpu().numpy().copy()
    output=WEEK/'results/validation/tactile'
    save_foot_snapshot(result,output,'standing_feet')
    save_snapshot(hands.sample_mujoco(data),output,'standing_hands')
    # Diagnostic pose only: lift the isolated CPU copy 10 mm without physics.
    data.qpos[2]+=.01
    mujoco.mj_forward(model,data)
    lifted=feet.sample_mujoco(data)
    lift_response=lifted.response.detach().cpu().numpy().copy()
    lift_points=lifted.tactile_points_w_per_sensor.detach().cpu().numpy().copy()
    save_foot_snapshot(lifted,output,'lifted_feet')
    assert foot_response.shape==(1,4,1024)
    assert foot_response.max()>0,'Standing soles should register proximity to floor'
    assert lift_response.max()==0,'Lifted soles must go dark'
    assert np.allclose(lift_points[...,:2],points[...,:2],atol=1e-6)
    assert np.allclose(lift_points[...,2]-points[...,2],.01,atol=1e-6)
    hashes={}
    for filename in ('flexitac.py','feet_tactile.py','tactile_display.py','mjlab_flexitac.py'):
        current=WEEK/'code/src/week3_tactile'/filename
        original=WEEK.parent/'Week3/code/src/sunday_a1'/filename
        # Locate the source package name from the Week3 tree, without changing it.
        if not original.exists():
            matches=list((WEEK.parent/'Week3/code').rglob(filename))
            if len(matches)!=1:raise ValueError(f'Cannot uniquely verify Week3 source {filename}')
            original=matches[0]
        assert original.read_bytes()==current.read_bytes(),f'Week3 sensor code changed: {filename}'
        hashes[filename]=hashlib.sha256(current.read_bytes()).hexdigest()
    report=dict(model_mass_kg=float(model.body_mass.sum()),duration_s=float(data.time),
                standing_response_max=float(foot_response.max()),lifted_response_max=float(lift_response.max()),
                foot_grid=[4,32,32],hand_grid=[4,12,32],source_unchanged=True,source_sha256=hashes,
                notes='Distance/proximity heatmaps, not calibrated pressure. Lift is an isolated diagnostic pose, not a walking result.')
    (output/'check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TACTILE_CHECK '+json.dumps(report),flush=True)


if __name__=='__main__':main()
