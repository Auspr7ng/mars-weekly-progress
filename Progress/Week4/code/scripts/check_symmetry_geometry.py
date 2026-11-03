"""Derive and check a left/right kinematic reflection from the actual CAD model."""
import bootstrap
from bootstrap import WEEK
import json,hashlib
import mujoco,numpy as np

m=mujoco.MjModel.from_xml_path(str(WEEK/'model/sunday_a1_locomotion.xml'))
d=mujoco.MjData(m);other=mujoco.MjData(m)
home=json.loads((WEEK/'model/arm_home.json').read_text());names=home['joint_order']
qadr=np.array([m.joint(n).qposadr[0] for n in names]);vadr=np.array([m.joint(n).dofadr[0] for n in names])
mirror=np.diag([1.,-1.,1.]);axial=-mirror
d.qpos[:]=m.qpos0;d.qpos[:3]=[0,0,home['root_height']];d.qpos[qadr]=0;mujoco.mj_forward(m,d)
indices=[];signs=[];alignments=[]
for n in names:
    mate=('l_'+n[2:]) if n.startswith('r_') else (('r_'+n[2:]) if n.startswith('l_') else n)
    a=m.joint(n);b=m.joint(mate);transform=mirror if a.type[0]==mujoco.mjtJoint.mjJNT_SLIDE else axial
    dot=float(d.xaxis[a.id]@(transform@d.xaxis[b.id]));indices.append(names.index(mate));signs.append(1 if dot>0 else -1);alignments.append(abs(dot))
indices=np.array(indices);signs=np.array(signs);assert min(alignments)>.99999
nominal=np.array([home['joint_pos'][n] for n in names]);offset=signs*nominal[indices]-nominal
sites=[m.site(s+'_sole').id for s in ('l','r')];rng=np.random.default_rng(20261112);position_error=orientation_error=torque_error=0.
for _ in range(100):
    d.qpos[:]=m.qpos0;d.qpos[:3]=[.1,.03,home['root_height']]
    q=nominal+rng.uniform(-.1,.1,len(names))
    for i,n in enumerate(names):
        if 'finger' in n:q[i]=.015+rng.uniform(-.003,.003)
    d.qpos[qadr]=q;mujoco.mj_forward(m,d)
    other.qpos[:]=d.qpos;other.qpos[:3]=mirror@d.qpos[:3];other.qpos[qadr]=signs*q[indices];mujoco.mj_forward(m,other)
    position_error=max(position_error,float(abs(other.site_xpos[sites]-d.site_xpos[sites[::-1]]@mirror).max()))
    ra=other.site_xmat[sites].reshape(2,3,3);rb=d.site_xmat[sites[::-1]].reshape(2,3,3)
    for column in (1,2):orientation_error=max(orientation_error,float(abs(ra[:,:,column]-rb[:,:,column]@mirror).max()))
    torque_error=max(torque_error,float(abs(other.qfrc_bias[vadr]-signs*d.qfrc_bias[vadr[indices]]).max()))
report=dict(joint_names=names,mirror_indices=indices.tolist(),mirror_signs=signs.tolist(),relative_position_offsets=offset.tolist(),minimum_axis_alignment=min(alignments),random_joint_poses=100,foot_position_max_error_m=position_error,foot_toe_normal_max_error=orientation_error,gravity_generalized_force_max_error=torque_error,passed=position_error<.0002 and orientation_error<.002 and torque_error<.02,notes='Offline kinematic/reflection check only. Approximate dynamics symmetry must still be validated in actual closed-loop training and both turn directions.')
report.update(exact_kinematic_symmetry=report['passed'],approximate_prior_usable=position_error<.002 and orientation_error<.002 and torque_error<.02,
              allowed_use='Soft mirror-consistency regularizer only; do not substitute reflected data for real physics rollouts.',
              model_sha256=hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest())
(WEEK/'results/validation/symmetry_geometry.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('joint_names','mirror_indices','mirror_signs','relative_position_offsets')},indent=2))
if report['approximate_prior_usable']:(WEEK/'config/mirror_map.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
