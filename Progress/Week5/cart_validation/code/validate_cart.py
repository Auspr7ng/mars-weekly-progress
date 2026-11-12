"""Independent cart validation only. No robot, policy training or task controller."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source/cart_v2.xml'
OUT = ROOT/'results'
CORNERS = ('fl','fr','rl','rr')


def build_scene(fork_mass=.05):
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    for corner in CORNERS:
        caster = root.find(f".//body[@name='caster_{corner}']")
        scale = fork_mass/.05
        ET.SubElement(caster, 'inertial', pos='.0075 0 -.0254', mass=str(fork_mass),
                      diaginertia=' '.join(str(v*scale) for v in (2e-5,2e-5,5e-6)))
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='validation_ground', type='plane', size='5 5 .1',
                  friction='.8 .01 .001', rgba='.88 .90 .93 1', condim='3')
    ET.SubElement(world, 'light', pos='0 -2 3', dir='0 1 -2', diffuse='.8 .8 .8')
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'texture', name='grid', type='2d', builtin='checker',
                  rgb1='.88 .90 .93', rgb2='.70 .74 .78', width='256', height='256')
    ET.SubElement(assets, 'material', name='ground_grid', texture='grid', texrepeat='40 40', texuniform='true')
    world.find("geom[@name='validation_ground']").set('material','ground_grid')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    # Fork inertials are assumptions for loading the test copy, not hardware data.
    root.insert(0, ET.Comment('VALIDATION COPY: four assumed fork masses/inertias; added floor/light/camera assets. Original source is unchanged.'))
    path = OUT/f'cart_test_fork_{fork_mass:g}kg.xml'
    ET.indent(tree, space='  ')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(path)), path


CASES = [
    dict(name='rest', duration=12., force=(0,0,0)),
    dict(name='coast_forward', duration=8., force=(0,0,0), initial_velocity=.2),
    dict(name='push_forward_5N', duration=8., force=(5,0,0)),
    dict(name='push_forward_10N', duration=8., force=(10,0,0)),
    dict(name='push_forward_20N', duration=8., force=(20,0,0)),
    dict(name='push_reverse_10N', duration=8., force=(-10,0,0)),
    dict(name='push_sideways_10N', duration=8., force=(0,10,0)),
    dict(name='push_diagonal_10N', duration=8., force=(math.sqrt(50),math.sqrt(50),0)),
    dict(name='casters_90deg_push10N', duration=8., force=(10,0,0), caster_angles=[math.pi/2]*4),
    dict(name='casters_mixed_push10N', duration=8., force=(10,0,0), caster_angles=[0,math.pi/2,math.pi,-math.pi/2]),
    dict(name='yaw_torque_0p4Nm', duration=8., force=(0,0,0), torque=(0,0,.4)),
]


def run_case(model, spec, renderer=None, writer=None):
    data = mujoco.MjData(model)
    cart = model.body('cart').id
    casters = [model.body('caster_'+c).id for c in CORNERS]
    wheels = [model.body('wheel_'+c).id for c in CORNERS]
    rolls = [model.joint('j_roll_'+c).dofadr[0] for c in CORNERS]
    swivels = [model.joint('j_swivel_'+c) for c in CORNERS]
    for joint, angle in zip(swivels, spec.get('caster_angles',[0]*4)):
        data.qpos[joint.qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135., -28., .9
    camera.lookat[:] = (0,0,.08)
    trace = []
    dt = model.opt.timestep
    initial_applied = False
    frame_stride = round(1/(25*dt))
    for step in range(round(spec['duration']/dt)):
        t = step*dt
        if t >= 2 and not initial_applied:
            if 'initial_velocity' in spec:
                data.qvel[0] = spec['initial_velocity']
                for roll in rolls:
                    data.qvel[roll] = spec['initial_velocity']/.0254
            initial_applied = True
        force = np.asarray(spec['force'],dtype=float) if 2 <= t < 4 else np.zeros(3)
        torque = np.asarray(spec.get('torque',(0,0,0)),dtype=float) if 2 <= t < 4 else np.zeros(3)
        point = data.xpos[cart]+data.xmat[cart].reshape(3,3)@np.array([-.21,0,.195])
        data.xfrc_applied[:] = 0.
        data.xfrc_applied[cart,:3] = force
        data.xfrc_applied[cart,3:] = torque+np.cross(point-data.xipos[cart],force)
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError(spec['name']+' nonfinite state')
        if step % 10 == 0:
            mujoco.mj_forward(model,data)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,cart,velocity,0)
            wheel_contacts = set()
            for ci in range(data.ncon):
                con = data.contact[ci]
                g1,g2 = int(con.geom1),int(con.geom2)
                if model.geom_type[g1] != mujoco.mjtGeom.mjGEOM_PLANE and model.geom_type[g2] != mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                wheel_contacts.update(b for b in (int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])) if b in wheels)
            errors = []
            for wheel, caster, roll in zip(wheels,casters,rolls):
                wv = np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,wheel,wv,0)
                along = float(wv[3:]@data.xmat[caster].reshape(3,3)[:,0])
                if wheel in wheel_contacts and max(abs(along),abs(.0254*data.qvel[roll])) > .01:
                    errors.append(abs(along-.0254*data.qvel[roll]))
            trace.append(dict(t=float(data.time),xyz=data.qpos[:3].tolist(),
                              linear_velocity=velocity[3:].tolist(),angular_velocity=velocity[:3].tolist(),
                              tilt_deg=math.degrees(math.acos(float(np.clip(data.xmat[cart].reshape(3,3)[2,2],-1,1)))),
                              wheel_contacts=len(wheel_contacts),
                              swivel_rad=[float(data.qpos[j.qposadr[0]]) for j in swivels],
                              roll_rad_s=[float(data.qvel[i]) for i in rolls],
                              rolling_slip_error_m_s=float(np.mean(errors)) if errors else None,
                              force_N=force.tolist()))
        if renderer is not None and writer is not None and step % frame_stride == 0:
            from PIL import Image, ImageDraw
            camera.lookat[:] = data.xpos[cart]+np.array([0,0,.08])
            renderer.update_scene(data,camera=camera)
            canvas = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0,0,960,76),fill='#152231')
            draw.text((12,10),'CART VALIDATION COPY | '+spec['name']+f' | t={data.time:.2f}s',fill='white')
            draw.text((12,31),'Assumed fork mass 0.05kg each; original source fails to compile.',fill='#ffce85')
            draw.text((12,52),f'External handle force {force[0]:+.1f}, {force[1]:+.1f} N | no robot/controller',fill='white')
            writer.send(np.asarray(canvas))
            if step == round(3/dt):
                canvas.save(OUT/(spec['name']+'.png'))
    xyz = np.array([r['xyz'] for r in trace])
    active = [r for r in trace if 3 <= r['t'] <= 4]
    settled = [r for r in trace if 1 <= r['t'] < 2]
    released = [r for r in trace if r['t'] > spec['duration']-1]
    summary = dict(name=spec['name'],duration_s=spec['duration'], finite=True,
                   warning_counts={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number},
                   total_mass_kg=float(model.body_mass.sum()),
                   max_tilt_deg=max(r['tilt_deg'] for r in trace),
                   xyz_displacement_m=(xyz[-1]-xyz[0]).tolist(),
                   rest_drift_m=float(np.linalg.norm(xyz[-1,:2]-xyz[0,:2])) if spec['name']=='rest' else None,
                   settled_wheel_contact_count_mean=float(np.mean([r['wheel_contacts'] for r in settled])),
                   forced_last_second_mean_velocity_m_s=np.mean([r['linear_velocity'] for r in active],axis=0).tolist(),
                   final_second_mean_speed_m_s=float(np.mean([np.linalg.norm(r['linear_velocity']) for r in released])),
                   max_speed_m_s=max(float(np.linalg.norm(r['linear_velocity'])) for r in trace),
                   max_vertical_speed_m_s=max(abs(r['linear_velocity'][2]) for r in trace),
                   max_yaw_speed_rad_s=max(abs(r['angular_velocity'][2]) for r in trace),
                   final_swivel_angles_deg=[math.degrees(v) for v in trace[-1]['swivel_rad']],
                   trace_file=spec['name']+'.json')
    (OUT/(spec['name']+'.json')).write_text(json.dumps(dict(case=spec,summary=summary,trace=trace),indent=2),encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        mujoco.MjModel.from_xml_path(str(SOURCE))
        original = dict(compiles=True)
    except ValueError as e:
        original = dict(compiles=False,error=str(e))
    model,scene = build_scene()
    results = []
    renderer = writer = None
    try:
        if args.video:
            import imageio_ffmpeg
            renderer = mujoco.Renderer(model,height=720,width=960)
            writer = imageio_ffmpeg.write_frames(str(OUT/'cart_validation.mp4'),(960,720),fps=25,codec='libx264',macro_block_size=1)
            writer.send(None)
        for spec in CASES:
            record = args.video and spec['name'] in ('rest','push_forward_10N','push_sideways_10N','casters_90deg_push10N')
            result = run_case(model,spec,renderer if record else None,writer if record else None)
            results.append(result)
            print(json.dumps(result),flush=True)
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    sensitivity = []
    for mass in (.01,.1):
        candidate,_ = build_scene(mass)
        spec = dict(CASES[3],name=f'fork_mass_{mass:g}_push10N')
        sensitivity.append(run_case(candidate,spec))
    summary = dict(mujoco_version=mujoco.__version__,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   original=original,validation_copy=str(scene.relative_to(ROOT)),
                   assumptions=dict(fork_mass_kg_each=.05,fork_inertia_kg_m2=[2e-5,2e-5,5e-6],
                                    fork_inertial_position_m=[.0075,0,-.0254],floor_friction=[.8,.01,.001],
                                    force_location='handle midpoint; 2 to 4 simulated seconds',
                                    no_robot_or_learning=True),
                   cases=results,mass_sensitivity=sensitivity,
                   note='Model validation only. Test-copy inertials and environment are assumptions, not measured hardware parameters.')
    (OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
