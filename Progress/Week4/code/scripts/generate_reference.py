"""Explore a slow IK gait as a fallback reference; never actuate the free root."""
import bootstrap
from bootstrap import WEEK
import json,math
import mujoco
import numpy as np
from scipy.optimize import least_squares


def main():
    home=json.loads((WEEK/'model/gait_home.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'))
    fk=mujoco.MjData(model)
    names=home['joint_order'];legs=home['leg_order']
    adr=np.array([model.joint(n).qposadr[0] for n in legs])
    alladr=np.array([model.joint(n).qposadr[0] for n in names])
    ids=[model.site(s+'_sole').id for s in ('l','r')]
    original=json.loads((WEEK/'model/home.json').read_text())
    fk.qpos[2]=original['root_height']
    for n,v in original['joint_pos'].items():fk.qpos[model.joint(n).qposadr[0]]=v
    mujoco.mj_forward(model,fk)
    foot0=fk.site_xpos[ids].copy();r0=fk.site_xmat[ids].copy()
    lower=np.array([model.jnt_range[model.joint(n).id,0] for n in legs]);upper=np.array([model.jnt_range[model.joint(n).id,1] for n in legs])
    middle=(lower+upper)/2
    lower=middle+.9*(lower-middle);upper=middle+.9*(upper-middle)
    baseq=np.array([home['joint_pos'][n] for n in names])
    samples=100;period=3.5;speed=.025;amplitude=.055;lift=.012
    reports=[];best=None
    for pelvis_pitch in (0.,.08,.14):
      for root_z in (.238,.232):
        poses=[];heights=[];errs=[];previous=np.array([home['joint_pos'][n] for n in legs])
        for index in range(samples):
            phase=index/samples
            root_y=amplitude*math.sin(2*math.pi*(phase-.1))
            target=foot0.copy()
            target[:,0]+=.013*(pelvis_pitch/.1)
            target[:,2]=.0004
            stride=speed*period*.7
            for foot,offset in ((0,0.),(1,.5)):
                p=(phase+offset)%1
                if p<.7:target[foot,0]+=stride/2-speed*period*p
                else:
                    u=(p-.7)/.3
                    smooth=u*u*(3-2*u)
                    target[foot,0]+=-stride/2+stride*smooth
                    target[foot,2]+=lift*math.sin(math.pi*u)
            def assign(q):
                fk.qpos[:]=model.qpos0
                fk.qpos[:3]=[0,root_y,root_z]
                fk.qpos[3:7]=[math.cos(pelvis_pitch/2),0,math.sin(pelvis_pitch/2),0]
                fk.qpos[alladr]=baseq;fk.qpos[adr]=q
                mujoco.mj_forward(model,fk)
            def error(q):
                assign(q)
                return np.r_[(fk.site_xpos[ids]-target).ravel()*150,
                             (fk.site_xmat[ids]-r0).ravel()*2,(q-previous)*.02]
            solved=least_squares(error,np.clip(previous,lower+1e-6,upper-1e-6),bounds=(lower,upper),max_nfev=150)
            previous=solved.x;assign(previous)
            q=baseq.copy()
            q[[names.index(n) for n in legs]]=previous
            poses.append(q)
            errs.append(float(np.max(np.linalg.norm(fk.site_xpos[ids]-target,axis=1))))
        poses=np.asarray(poses)
        report=dict(pelvis_pitch=pelvis_pitch,root_height=root_z,max_ik_error_mm=max(errs)*1000)
        data=mujoco.MjData(model)
        data.qpos[2]=root_z;data.qpos[3:7]=[math.cos(pelvis_pitch/2),0,math.sin(pelvis_pitch/2),0]
        data.qpos[alladr]=poses[10]
        data.ctrl[:]=[poses[10,names.index(model.actuator(i).name)] for i in range(model.nu)]
        failed=False;trace=[];max_lift=np.zeros(2)
        for step in range(round((2+3*period)/model.opt.timestep)):
            elapsed=max(0.,data.time-2.)
            p=(.1+elapsed/period)%1
            j=p*samples;i=int(j)%samples;f=j-int(j)
            q=(1-f)*poses[i]+f*poses[(i+1)%samples]
            data.ctrl[:]=[q[names.index(model.actuator(a).name)] for a in range(model.nu)]
            mujoco.mj_step(model,data)
            if step%10==0:
                mujoco.mj_forward(model,data)
                rot=data.site_xmat[ids].reshape(2,3,3)
                low=data.site_xpos[ids,2]-np.abs(rot[:,2,0])*.0525-np.abs(rot[:,2,1])*.09
                max_lift=np.maximum(max_lift,low)
                tilt=math.acos(np.clip(data.xmat[model.body('base_link').id].reshape(3,3)[2,2],-1,1))
                trace.append([float(data.time),*data.qpos[:3].tolist(),tilt,*low.tolist()])
                if tilt>.8 or data.qpos[2]<.15:failed=True;break
        forward=float(data.qpos[0]);side=float(data.qpos[1])
        good=bool(not failed and forward>.12 and abs(side)<.15 and min(max_lift)>.005 and max(errs)<.004)
        report.update(native_survived=not failed,forward_m=forward,lateral_m=side,max_sole_clearance=max_lift.tolist(),candidate_passed=good)
        reports.append(report);print(json.dumps(report),flush=True)
        if good and (best is None or abs(forward-speed*3*period)<best[0]):best=(abs(forward-speed*3*period),poses,pelvis_pitch,root_z,trace)
    output=WEEK/'research/reference_candidate';output.mkdir(exist_ok=True)
    (output/'report.json').write_text(json.dumps(dict(candidates=reports,notes='Offline IK and native PD baseline only; no learned result. Root is set only on reset, then evolves under physics.'),indent=2),encoding='utf-8')
    if best:
        _,poses,pitch,z,trace=best
        np.savez_compressed(output/'gait.npz',joint_pos=poses,joint_names=np.array(names),period=period,speed=speed,pelvis_pitch=pitch,root_height=z,trace=np.asarray(trace))
        print('REFERENCE_CANDIDATE_SAVED',flush=True)
    else:print('No candidate passed; none used for training.',flush=True)


if __name__=='__main__':main()
