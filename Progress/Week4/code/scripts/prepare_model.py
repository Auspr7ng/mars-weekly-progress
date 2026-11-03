"""Build a floating-base model preserving Week3 geometry and link frames.

The extra torso inertia and actuator parameters are explicit SIMULATION
ASSUMPTIONS, not measurements or verified manufacturer specifications.
"""
import bootstrap
from bootstrap import WEEK
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares


def fmt(value):
    return " ".join(f"{float(x):.10g}" for x in value)


def actuator_params(name):
    if "finger" in name:
        return dict(kp=700.0, kd=15.0, effort=15.0, speed=0.05, armature=0.001, scale=0.0)
    if any(k in name for k in ("hip", "thigh", "knee", "ankle")):
        return dict(kp=50.0, kd=1.0, effort=1.6, speed=4.0, armature=0.005, scale=0.20)
    return dict(kp=50.0 if "wrist" not in name else 30.0, kd=1.0, effort=.5 if name=='head_rotation' else (1.6 if "wrist" not in name else .8),
                speed=4.0, armature=0.005, scale=0.10 if "wrist" not in name else 0.06)


def build():
    profile=json.loads((WEEK/'config/dynamics_profile.json').read_text())
    source = WEEK / "model/source/week3_fixed_base.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "Sunday_A1_Week4_SIM_ASSUMPTIONS")
    root.find("compiler").set("meshdir", "meshes")
    root.find("compiler").set("fusestatic", "false")
    original = mujoco.MjModel.from_xml_path(str(WEEK.parent / "Week3/code/src/sunday_a1_mjlab/assets/sunday_a1.xml"))
    world = root.find("worldbody")
    cad = world.find("body[@name='base_link']")
    cad.set("name", "cad_base_link")
    world.remove(cad)
    # Canonical +X forward maps to CAD -Y. All original link frames are retained.
    floating = ET.SubElement(world, "body", name="base_link")
    ET.SubElement(floating, "freejoint", name="floating_base")
    torso_mass = profile['torso']['mass_kg']
    ET.SubElement(floating, "inertial", mass=str(torso_mass), pos=fmt(profile['torso']['com_canonical_m']),
                  fullinertia=fmt(profile['torso']['fullinertia_canonical_kg_m2']))
    cad.set("quat", fmt([math.sqrt(0.5), 0, 0, math.sqrt(0.5)]))
    floating.append(cad)
    contact = ET.SubElement(root, "contact")
    parent = {}
    bodies = {b.get("name"): b for b in root.findall(".//body")}
    for b in root.findall(".//body"):
        for child in b.findall("body"):
            parent[child.get("name")] = b.get("name")
    proxy_report = []
    urdf=ET.parse(WEEK/'model/source/sunday_a1.urdf').getroot()
    source_inertials={link.get('name'):link.find('inertial') for link in urdf.findall('link')}
    for bname, body in bodies.items():
        if bname in source_inertials and bname!='base_link':
            src=source_inertials[bname]
            if any(float(v)!=0 for v in src.find('origin').get('rpy','0 0 0').split()):
                raise ValueError('Nonzero inertial RPY requires explicit tensor transformation')
            dst=body.find('inertial');a=src.find('inertia').attrib
            dst.attrib.clear()
            factor=profile['global_mass_inertia_scale']
            dst.set('mass',str(float(src.find('mass').get('value'))*factor))
            dst.set('pos',src.find('origin').get('xyz'))
            dst.set('fullinertia',fmt([float(a[k])*factor for k in ['ixx','iyy','izz','ixy','ixz','iyz']]))
        old_name = "base_link" if bname == "cad_base_link" else bname
        colliders = [g for g in body.findall("geom") if g.get("contype", "1") != "0"]
        for g in colliders:
            body.remove(g)
        for g in body.findall("geom"):
            g.set("contype", "0"); g.set("conaffinity", "0"); g.set("group", "1")
        if not colliders:
            continue
        if "foot_link" in bname:
            center = np.array([0.005 if bname.startswith("l_") else -0.005, -0.051175, -0.025])
            half = np.array([0.0525, 0.09, 0.005])
        elif 'finger_' in bname:
            pad_y={'l_finger_bottom_1':-.0494,'l_finger_top_1':.0289,
                   'r_finger_bottom_1':.0488,'r_finger_top_1':-.0309}
            center=np.array([-.134,pad_y[bname],-.035])
            half=np.array([.012,.0005,.032])
        else:
            points = []
            for gid in range(original.ngeom):
                if original.body(int(original.geom_bodyid[gid])).name != old_name or original.geom_contype[gid] == 0:
                    continue
                meshid = original.geom_dataid[gid]
                start, count = original.mesh_vertadr[meshid], original.mesh_vertnum[meshid]
                rot = np.zeros(9); mujoco.mju_quat2Mat(rot, original.geom_quat[gid])
                points.append(original.mesh_vert[start:start+count] @ rot.reshape(3, 3).T + original.geom_pos[gid])
            points = np.concatenate(points)
            lower, upper = points.min(0), points.max(0)
            center = (lower+upper)/2
            half = np.maximum((upper-lower)/2 * 0.82, 0.002)
        ET.SubElement(body, "geom", name="collision_"+bname, type="box", pos=fmt(center), size=fmt(half),
                      contype="1", conaffinity="1", group="3", mass="0", rgba="0.3 0.7 0.4 0.25",
                      friction="0.8 0.005 0.0001", condim="3", solref="0.01 1")
        proxy_report.append(dict(body=bname,center=center.tolist(),half_size=half.tolist()))
    # Exclude only immediate/grandparent mechanical housings and same gripper fingers.
    excluded=set()
    for child,p in parent.items():
        for ancestor in [p, parent.get(p)]:
            if ancestor is not None:
                pair=tuple(sorted([child,ancestor])); excluded.add(pair)
    for side in ("l", "r"):
        excluded.add((f"{side}_finger_bottom_1",f"{side}_finger_top_1"))
    for a,b in sorted(excluded):
        ET.SubElement(contact,"exclude",body1=a,body2=b)
    for side in ("l", "r"):
        body = bodies[f"{side}_foot_link_1"]
        ET.SubElement(body,"site",name=f"{side}_sole",pos=fmt([0.005 if side=="l" else -0.005,-0.051175,-0.03]),size="0.003")
    params={}
    actuators=ET.SubElement(root,"actuator")
    for joint in root.findall(".//joint"):
        name=joint.get("name")
        p=actuator_params(name); params[name]=p
        joint.set("actuatorfrcrange",fmt([-p['effort'],p['effort']]))
        joint.set("armature",str(p['armature']))
        joint.set("damping","0.1" if "finger" not in name else "0.01")
        if name=="head_rotation":
            joint.set("range","-1.2 1.2");joint.set("limited","true")
        ET.SubElement(actuators,"position",name=name,joint=name,kp=str(p['kp']),kv=str(p['kd']),
                      forcerange=fmt([-p['effort'],p['effort']]))
    ET.indent(root)
    model_path=WEEK/'model/sunday_a1_locomotion.xml'
    tree.write(model_path,encoding='utf-8',xml_declaration=True)
    # Native diagnostic scene uses exactly this robot plus a ground plane.
    scene=ET.Element('mujoco',model='Sunday_A1_Week4_native')
    ET.SubElement(scene,'include',file='sunday_a1_locomotion.xml')
    ET.SubElement(scene,'option',timestep='0.002',integrator='implicitfast',iterations='30',cone='elliptic')
    sw=ET.SubElement(scene,'worldbody')
    ET.SubElement(sw,'geom',name='ground',type='plane',size='20 20 .1',rgba='.2 .23 .28 1',friction='.8 .005 .0001')
    ET.SubElement(sw,'light',pos='0 -2 3',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.indent(scene)
    ET.ElementTree(scene).write(WEEK/'model/scene.xml',encoding='utf-8')
    model=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'))
    data=mujoco.MjData(model)
    joint_names=[model.joint(i).name for i in range(1,model.njnt)]
    q0={name:(0.015 if 'finger' in name else 0.0) for name in joint_names}
    legs=[n for n in joint_names if any(k in n for k in ('hip','thigh','knee','ankle'))]
    pref={n:0.0 for n in legs}
    pref.update(l_hip_roll=.15,r_hip_roll=-.15,l_knee_roll=.20,r_knee_roll=-.20,
                l_ankle_roll=-.085,r_ankle_roll=-.085,l_thigh_yaw=-.096,r_thigh_yaw=.096)
    ids=[model.joint(n).qposadr[0] for n in legs]
    soleids=[model.site(f'{s}_sole').id for s in ('l','r')]
    def assign(x):
        data.qpos[:]=model.qpos0
        data.qpos[2]=x[-1]
        for n,v in q0.items():data.qpos[model.joint(n).qposadr[0]]=v
        data.qpos[ids]=x[:-1]
        mujoco.mj_forward(model,data)
    assign(np.r_[[0]*12,.25])
    initial_xy=data.site_xpos[soleids,:2].copy()
    def residual(x):
        assign(x)
        terms=[(data.site_xpos[soleids,2]-0.0003)*100,
               (data.site_xpos[soleids,:2]-initial_xy).ravel()*2,
               (data.site_xpos[soleids,:2].mean(0)-data.subtree_com[model.body('base_link').id,:2])*40]
        for sid in soleids:
            R=data.site_xmat[sid].reshape(3,3)
            # Sole up and sole headings preserve the canonical CAD orientation.
            terms.append(R[:2,2]*4)
            terms.append(np.array([R[0,0]])*2)
        terms.append((x[:-1]-np.array([pref[n] for n in legs]))*.1)
        return np.concatenate(terms)
    lo=np.array([model.jnt_range[model.joint(n).id,0]+.03 for n in legs]+[.15])
    hi=np.array([model.jnt_range[model.joint(n).id,1]-.03 for n in legs]+[.30])
    fit=least_squares(residual,np.r_[[pref[n] for n in legs],.247],bounds=(lo,hi),max_nfev=500)
    assign(fit.x)
    for n,v in zip(legs,fit.x[:-1]):q0[n]=float(v)
    # Action order follows model joint definition order, explicitly saved.
    arms=[n for n in joint_names if any(k in n for k in ('shoulder','elbow','wrist'))]
    home=dict(root_height=float(fit.x[-1]),joint_pos=q0,joint_order=joint_names,leg_order=legs,arm_order=arms,
              actor_joint_count=len(joint_names),actuators=params,sole_sites=['l_sole','r_sole'],
              total_mass_kg=float(model.body_mass.sum()),simulation_only=True,hardware_calibrated=False)
    (WEEK/'model/home.json').write_text(json.dumps(home,indent=2),encoding='utf-8')
    if abs(home['total_mass_kg']-profile['total_estimated_mass_kg'])>1e-8:
        raise ValueError('Compiled robot does not match measured-total mass budget')
    assumption=dict(status='2.84 KG MEASURED-TOTAL BUDGET; PER-LINK INERTIA ESTIMATED; ST3215 AT 7.2 V',source='Modified geometry retained; all mass/inertia tensors normalized together. Root template is included within total, not added above it.',
                    global_mass_inertia_scale=profile['global_mass_inertia_scale'],battery_included=False,
                    added_root_mass_kg=torso_mass,root_com_canonical_m=profile['torso']['com_canonical_m'],
                    root_fullinertia_canonical_kg_m2=profile['torso']['fullinertia_canonical_kg_m2'],reason='See config/dynamics_profile.json for exact sources and modified-gripper assumptions',
                    canonical_frame='CAD -Y forward rotated +90 deg about Z; original child link frames retained',
                    collision_proxies=proxy_report,excluded_pairs=sorted(excluded),actuators=params,
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    (WEEK/'config/simulation_assumptions.json').write_text(json.dumps(assumption,indent=2),encoding='utf-8')
    data.ctrl[:]=[q0[model.actuator(i).name] for i in range(model.nu)]
    initial_contacts=[dict(a=model.geom(c.geom1).name,b=model.geom(c.geom2).name,distance=float(c.dist)) for c in data.contact]
    trace=[]
    for step in range(5000):
        mujoco.mj_step(model,data)
        if step%50==0:
            trace.append([data.time,*data.qpos[:3],*data.qpos[3:7]])
    tilt=float(np.arccos(np.clip(data.xmat[model.body('base_link').id].reshape(3,3)[2,2],-1,1)))
    report=dict(nq=model.nq,nv=model.nv,nu=model.nu,nmocap=model.nmocap,total_mass_kg=home['total_mass_kg'],
                h0=home['root_height'],home=q0,fit_residual=float(np.linalg.norm(fit.fun)),initial_contacts=initial_contacts,
                passive_pd_duration_s=data.time,final_root_xyz=data.qpos[:3].tolist(),final_tilt_deg=math.degrees(tilt),
                finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
                passive_pd_standing=bool(data.qpos[2]>.7*home['root_height'] and tilt<.3),trace=trace)
    (WEEK/'results/validation/native_model.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['trace','home','initial_contacts']},indent=2))
    print('Initial contacts:',json.dumps(initial_contacts))


if __name__=='__main__':
    build()
