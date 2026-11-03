"""Verify whole-sole peak clearance at a planned weight-transfer pose."""
import bootstrap
from bootstrap import WEEK
import argparse,json,math
import mujoco,numpy as np
from scipy.optimize import least_squares

p=argparse.ArgumentParser();p.add_argument('--pose',default='natural_home');args=p.parse_args()
h=json.loads((WEEK/f'model/{args.pose}.json').read_text());m=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'));d=mujoco.MjData(m)
names=h['leg_order'];adr=[m.joint(n).qposadr[0] for n in names];q0=np.array([h['joint_pos'][n] for n in names])
lo=np.array([m.jnt_range[m.joint(n).id,0] for n in names]);hi=np.array([m.jnt_range[m.joint(n).id,1] for n in names]);mid=(lo+hi)/2
scale=np.array([.65 if 'knee' in n else (.45 if 'roll' in n else .35) for n in names])
lo=np.maximum(mid+.9*(lo-mid),q0-scale);hi=np.minimum(mid+.9*(hi-mid),q0+scale)
sites=[m.site(s+'_sole').id for s in ('l','r')]
def assign(q,z,y,pitch):
    d.qpos[:]=m.qpos0;d.qpos[:3]=[0,y,z];d.qpos[3:7]=[math.cos(pitch/2),0,math.sin(pitch/2),0]
    for n,v in h['joint_pos'].items():d.qpos[m.joint(n).qposadr[0]]=v
    d.qpos[adr]=q;mujoco.mj_forward(m,d)
assign(q0,h['root_height'],0,0);base=d.site_xpos[sites].copy();rot=d.site_xmat[sites].copy()
reports=[]
for z in (.240,.238,.236):
 for pitch in (0.,.05):
  for swing in (0,1):
    target=base.copy();target[:,2]=.0004;target[swing,2]+=.020
    def error(q):
        assign(q,z,-.028 if swing==0 else .028,pitch)
        return np.r_[(d.site_xpos[sites]-target).ravel()*200,(d.site_xmat[sites]-rot).ravel()*12,(q-q0)*.01]
    sol=least_squares(error,np.clip(q0,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=600)
    assign(sol.x,z,-.028 if swing==0 else .028,pitch)
    r=d.site_xmat[sites].reshape(2,3,3);low=d.site_xpos[sites,2]-np.abs(r[:,2,0])*.0525-np.abs(r[:,2,1])*.09
    reports.append(dict(root_z=z,body_pitch=pitch,swing=['left','right'][swing],max_position_error_mm=float(np.linalg.norm(d.site_xpos[sites]-target,axis=1).max()*1000),whole_sole_clearance_mm=(low*1000).tolist()))
(WEEK/f'results/quality/swing_reach_{args.pose.removesuffix("_home")}.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print(json.dumps(reports,indent=2))
