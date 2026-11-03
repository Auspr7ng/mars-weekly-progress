"""Check feasible sole motion and prepare a lower walking stance, without root actuation."""
import bootstrap
from bootstrap import WEEK
import json
import math
import mujoco
import numpy as np
from scipy.optimize import least_squares


def main():
    home=json.loads((WEEK/'model/home.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'))
    data=mujoco.MjData(model)
    legs=home['leg_order']
    adr=np.array([model.joint(n).qposadr[0] for n in legs])
    q0=np.array([home['joint_pos'][n] for n in legs])
    lo=np.array([model.jnt_range[model.joint(n).id,0] for n in legs])
    hi=np.array([model.jnt_range[model.joint(n).id,1] for n in legs])
    mid=(lo+hi)/2
    lo,hi=mid+.9*(lo-mid),mid+.9*(hi-mid)
    sites=[model.site(s+'_sole').id for s in ('l','r')]
    def assign(q,z):
        data.qpos[:]=model.qpos0
        data.qpos[2]=z
        for n,v in home['joint_pos'].items():data.qpos[model.joint(n).qposadr[0]]=v
        data.qpos[adr]=q
        mujoco.mj_forward(model,data)
    assign(q0,home['root_height'])
    initial=data.site_xpos[sites].copy()
    initial_rot=data.site_xmat[sites].copy()
    def fit(q,z,target,bounds,reference,balance=False):
        def residual(x):
            assign(x,z)
            weights=np.array([10,20,300]) if balance else np.array([100,100,100])
            balance_error=(data.subtree_com[model.body('base_link').id,:2]-data.site_xpos[sites,:2].mean(axis=0))*100 if balance else np.zeros(2)
            return np.r_[((data.site_xpos[sites]-target)*weights).ravel(),balance_error,
                         (data.site_xmat[sites]-initial_rot).ravel()*1,
                         (x-reference)*.02]
        result=least_squares(residual,np.clip(q,bounds[0]+1e-7,bounds[1]-1e-7),bounds=bounds,max_nfev=1000)
        assign(result.x,z)
        return result.x,dict(max_position_error_mm=float(np.linalg.norm(data.site_xpos[sites]-target,axis=1).max()*1000),
                             sole_heights_m=data.site_xpos[sites,2].tolist(),joint_target=dict(zip(legs,result.x.tolist())))
    target=initial.copy();target[0,0]+=.025;target[0,2]+=.018
    _,old=fit(q0,home['root_height'],target,(np.maximum(lo,q0-.2),np.minimum(hi,q0+.2)),q0)
    z=home['root_height']-.005
    q=q0.copy()
    q[legs.index('r_knee_roll')]=-.5
    q[legs.index('l_knee_roll')]=.5
    q,stance=fit(q,z,initial,(lo,hi),q,balance=True)
    position_error=float(np.abs(data.site_xpos[sites,2]-initial[:,2]).max()*1000)
    for n,v in zip(legs,q):home['joint_pos'][n]=float(v)
    home['root_height']=z
    home['walking_revision']=2
    home['source']='Same physical model, IK lower stance; no mass, motor or geometry changes'
    assign(q,z)
    data.ctrl[:]=[home['joint_pos'][model.actuator(i).name] for i in range(model.nu)]
    peak_torque=0.
    for _ in range(5000):
        mujoco.mj_step(model,data)
        peak_torque=max(peak_torque,float(np.max(np.abs(data.actuator_force))))
    tilt=math.degrees(math.acos(np.clip(data.xmat[model.body('base_link').id].reshape(3,3)[2,2],-1,1)))
    drift=float(np.linalg.norm(data.qpos[:2]))
    good=bool(position_error<1 and tilt<10 and drift<.02 and data.qpos[2]>.9*z)
    home['native_pd_verified']=good
    if good:(WEEK/'model/gait_home.json').write_text(json.dumps(home,indent=2),encoding='utf-8')
    target=initial.copy();target[0,0]+=.025;target[0,2]+=.018
    _,new=fit(q,z,target,(np.maximum(lo,q-.55),np.minimum(hi,q+.55)),q)
    report=dict(original_action_range_step=old,lower_stance=stance,lower_root_height_m=z,
                lower_stance_native_pd=dict(passed=good,tilt_deg=tilt,drift_m=drift,peak_actuator_force=peak_torque),
                proposed_action_range_step=new,notes='Kinematic feasibility and gravity/PD check only. Not a walking controller or learned result.')
    (WEEK/'results/validation/step_range.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not good:raise RuntimeError('Proposed stance did not pass native PD verification')


if __name__=='__main__':main()
