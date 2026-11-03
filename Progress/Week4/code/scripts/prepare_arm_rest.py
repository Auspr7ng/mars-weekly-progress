"""Prepare a bent-elbow rest pose and rebalance stance with unchanged inertias."""
import bootstrap
from bootstrap import WEEK
import json,math
import mujoco,numpy as np
from scipy.optimize import least_squares

h=json.loads((WEEK/'model/natural_home.json').read_text());m=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'));d=mujoco.MjData(m)
legs=h['leg_order'];adr=[m.joint(n).qposadr[0] for n in legs];q0=np.array([h['joint_pos'][n] for n in legs])
for s in ('l','r'):h['joint_pos'][s+'_elbow_pitch']=.70
def assign(q):
    d.qpos[:]=m.qpos0;d.qpos[2]=h['root_height']
    for n,v in h['joint_pos'].items():d.qpos[m.joint(n).qposadr[0]]=v
    d.qpos[adr]=q;mujoco.mj_forward(m,d)
assign(q0);sites=[m.site(s+'_sole').id for s in ('l','r')]
target=d.site_xpos[sites].copy();target[:,0]+=d.subtree_com[m.body('base_link').id,0]-target[:,0].mean();target[:,2]=.0004
rotation=d.site_xmat[sites].copy()
lo=np.array([m.jnt_range[m.joint(n).id,0] for n in legs]);hi=np.array([m.jnt_range[m.joint(n).id,1] for n in legs]);mid=(lo+hi)/2;lo=mid+.9*(lo-mid);hi=mid+.9*(hi-mid)
hi[legs.index('l_knee_roll')]=-.05;lo[legs.index('r_knee_roll')]=.05
def residual(q):
    assign(q)
    return np.r_[(d.site_xpos[sites]-target).ravel()*200,(d.site_xmat[sites]-rotation).ravel()*3,
                 (d.subtree_com[m.body('base_link').id,:2]-d.site_xpos[sites,:2].mean(0))*80,(q-q0)*.01]
sol=least_squares(residual,np.clip(q0,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=600);assign(sol.x)
for n,v in zip(legs,sol.x):h['joint_pos'][n]=float(v)
h['sole_center_xy_root']=d.site_xpos[sites,:2].tolist();h['walking_revision']=7
d.ctrl[:]=[h['joint_pos'][m.actuator(a).name] for a in range(m.nu)]
for _ in range(5000):mujoco.mj_step(m,d)
mujoco.mj_forward(m,d)
tilt=math.degrees(math.acos(np.clip(d.xmat[m.body('base_link').id].reshape(3,3)[2,2],-1,1)));drift=float(np.linalg.norm(d.qpos[:2]))
report=dict(tilt_deg=tilt,drift_m=drift,foot_target=target.tolist(),passed=bool(tilt<3 and drift<.02 and d.qpos[2]>.9*h['root_height']))
(WEEK/'results/quality/arm_rest_stance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if not report['passed']:raise RuntimeError(json.dumps(report))
h['source']='Correct flexion branch, bent elbows, stance balanced for new arm pose. No mass/inertia/torque changes.'
(WEEK/'model/arm_home.json').write_text(json.dumps(h,indent=2),encoding='utf-8')
print(json.dumps(report))
