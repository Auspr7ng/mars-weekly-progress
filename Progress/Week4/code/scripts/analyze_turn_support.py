"""Offline reachability/static-support diagnostic; never used to drive a rollout."""
import bootstrap
from bootstrap import WEEK
import argparse,json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--knee-scale',type=float,default=.65);p.add_argument('--ankle-scale',type=float,default=.45)
p.add_argument('--roll',type=float,default=0.);p.add_argument('--clearance',type=float,default=.024);p.add_argument('--output',default='turn_target_static_support.json');args=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(WEEK/'model/sunday_a1_locomotion.xml'))
d=mujoco.MjData(m);home=json.loads((WEEK/'model/arm_home.json').read_text())
legs=home['leg_order'];qa=np.array([m.joint(n).qposadr[0] for n in legs]);va=np.array([m.joint(n).dofadr[0] for n in legs])
sites=[m.site(s+'_sole').id for s in ('l','r')];root=m.body('base_link').id
q0=m.qpos0.copy();q0[:3]=[0,0,home['root_height']]
for name,value in home['joint_pos'].items():q0[m.joint(name).qposadr[0]]=value
d.qpos[:]=q0;mujoco.mj_forward(m,d)
positions=d.site_xpos[sites].copy();rotations=d.site_xmat[sites].reshape(2,3,3).copy()
scales=np.array([args.knee_scale if 'knee' in n else (args.ankle_scale if 'ankle_roll' in n else (.45 if 'roll' in n else .35)) for n in legs])
ranges=np.array([m.joint(n).range for n in legs]);mid=ranges.mean(axis=1);half=np.diff(ranges,axis=1).ravel()*.45
bounds=(np.maximum(q0[qa]-scales,mid-half),np.minimum(q0[qa]+scales,mid+half));reports=[]
for amplitude in (.018,.028,.035):
    for swing in (0,1):
        target=positions.copy();target[swing,2]+=args.clearance
        d.qpos[:]=q0;d.qpos[1]=amplitude*(-1 if swing==0 else 1)
        roll=args.roll*(1 if swing==0 else -1);d.qpos[3:7]=[np.cos(roll/2),np.sin(roll/2),0,0]
        def error(q):
            d.qpos[qa]=q;mujoco.mj_forward(m,d)
            pos=(d.site_xpos[sites]-target).ravel()/.01
            actual=d.site_xmat[sites].reshape(2,3,3)
            angle=Rotation.from_matrix(actual@rotations.transpose(0,2,1)).as_rotvec().ravel()/.1
            return np.r_[pos,angle]
        opt=least_squares(error,q0[qa],bounds=bounds,xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=300)
        error(opt.x);stance=1-swing;sid=sites[stance];r=d.site_xmat[sid].reshape(3,3)
        com=d.subtree_com[root].copy();offset=r.T@(com-d.site_xpos[sid])
        margin=float(np.min(np.array([.0525,.09])-np.abs(offset[:2])))
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,sid);jac=np.r_[jp,jr]
        wrench=np.linalg.solve(jac[:,:6].T,d.qfrc_bias[:6]);tau=d.qfrc_bias-jac.T@wrench
        reports.append(dict(root_sway_m=amplitude,swing=('left','right')[swing],position_error_max_mm=float(np.abs(d.site_xpos[sites]-target).max()*1000),orientation_error_max_rad=float(np.linalg.norm(Rotation.from_matrix(d.site_xmat[sites].reshape(2,3,3)@rotations.transpose(0,2,1)).as_rotvec(),axis=1).max()),whole_robot_com_m=com.tolist(),stance_support_margin_mm=margin*1000,peak_required_leg_static_torque_Nm=float(np.abs(tau[va]).max()),limiting_joint=legs[int(np.abs(tau[va]).argmax())],action_bounds_active=[legs[i] for i in np.flatnonzero(opt.active_mask)],notes='Static IK diagnostic with both feet flat and one raised24mm. Root pose is prescribed only inside this offline calculation. Not a physical rollout or proof of dynamic stability.'))
for report in reports:report.update(knee_scale=args.knee_scale,ankle_roll_scale=args.ankle_scale,root_roll_amplitude_rad=args.roll,target_clearance_m=args.clearance)
out=WEEK/'results/validation'/args.output;out.write_text(json.dumps(reports,indent=2),encoding='utf-8')
for x in reports:print({k:x[k] for k in ('root_sway_m','swing','position_error_max_mm','stance_support_margin_mm','peak_required_leg_static_torque_Nm','limiting_joint','action_bounds_active')})
