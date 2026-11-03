"""Validate a less wide stance and swing reach; physical masses/limits unchanged."""
import bootstrap
from bootstrap import WEEK
import json,math
import mujoco,numpy as np
from scipy.optimize import least_squares

home=json.loads((WEEK/'model/gait_home.json').read_text())
m=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'));d=mujoco.MjData(m)
names=home['leg_order'];adr=[m.joint(n).qposadr[0] for n in names]
lower=np.array([m.jnt_range[m.joint(n).id,0] for n in names]);upper=np.array([m.jnt_range[m.joint(n).id,1] for n in names]);mid=(lower+upper)/2
lower=mid+.9*(lower-mid);upper=mid+.9*(upper-mid)
sites=[m.site(s+'_sole').id for s in ('l','r')]
def assign(q,z):
    d.qpos[:]=m.qpos0;d.qpos[2]=z
    for n,v in home['joint_pos'].items():d.qpos[m.joint(n).qposadr[0]]=v
    d.qpos[adr]=q;mujoco.mj_forward(m,d)
initial=np.array([home['joint_pos'][n] for n in names]);assign(initial,home['root_height'])
target=d.site_xpos[sites].copy();target[:,1]=[.070,-.070];target[:,2]=.0004
# Flat soles, with their CAD heading preserved.
old_rotation=d.site_xmat[sites].reshape(2,3,3)
yaws=np.arctan2(old_rotation[:,1,0],old_rotation[:,0,0])
orientation=np.array([[[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]] for y in yaws])
reports=[];chosen=None
initial[names.index('l_knee_roll')]=-.70
initial[names.index('r_knee_roll')]=.70
initial[names.index('l_hip_roll')]=-.35
initial[names.index('r_hip_roll')]=.35
initial[names.index('l_ankle_roll')]=.40
initial[names.index('r_ankle_roll')]=.40
for z in (.240,.238,.236):
    def residual(q):
        assign(q,z)
        return np.r_[(d.site_xpos[sites]-target).ravel()*200,
                     (d.site_xmat[sites].reshape(2,3,3)-orientation).ravel()*3,
                     (d.subtree_com[m.body('base_link').id,:2]-d.site_xpos[sites,:2].mean(0))*60,(q-initial)*.015]
    solved=least_squares(residual,np.clip(initial,lower+1e-6,upper-1e-6),bounds=(lower,upper),max_nfev=800)
    q=solved.x;assign(q,z)
    qfull={**home['joint_pos'],**dict(zip(names,q.tolist()))}
    error=float(np.linalg.norm(d.site_xpos[sites]-target,axis=1).max())
    rotation=d.site_xmat[sites].reshape(2,3,3);tilt_soles=np.degrees(np.arccos(np.clip(rotation[:,2,2],-1,1)))
    d.ctrl[:]=[qfull[m.actuator(i).name] for i in range(m.nu)]
    d.qvel[:]=0
    for _ in range(5000):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    tilt=float(np.degrees(np.arccos(np.clip(d.xmat[m.body('base_link').id].reshape(3,3)[2,2],-1,1))))
    drift=float(np.linalg.norm(d.qpos[:2]));good=bool(error<.001 and tilt<5 and drift<.02 and d.qpos[2]>.9*z)
    report=dict(root_height=z,ik_error_mm=error*1000,initial_sole_tilt_deg=tilt_soles.tolist(),native_10s_tilt_deg=tilt,native_drift_m=drift,passed=good)
    reports.append(report);print(json.dumps(report),flush=True)
    if good and chosen is None:chosen=(qfull,z)
if chosen is None:raise RuntimeError('No quality stance passed native validation')
qfull,z=chosen
assign(np.array([qfull[n] for n in names]),z)
home['sole_center_xy_root']=d.site_xpos[sites,:2].tolist()
home.update(joint_pos=qfull,root_height=z,walking_revision=6,soft_joint_pos_limit_factor=.9,
            source='Opposite knee-flexion IK branch with positive ankle angles; 140mm foot-center spacing; native PD validated, physical parameters unchanged')
(WEEK/'model/natural_home.json').write_text(json.dumps(home,indent=2),encoding='utf-8')
(WEEK/'results/quality').mkdir(exist_ok=True)
(WEEK/'results/quality/natural_stance.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
