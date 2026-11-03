"""Additional diagnostics from measured rollout traces, independent of rewards."""
import numpy as np


def detailed_metrics(a,mass,period=1.4,offset=.075,actuator_names=None,force_limits=None):
    n=len(a['xyz']);time=(np.arange(n)+1)*.02;warm=min(100,n//4)
    cycle=time/period+offset
    variable='gait_phase' in a
    if variable:cycle=np.unwrap(np.asarray(a['gait_phase'])*2*np.pi)/(2*np.pi)
    ph=np.stack([cycle%1,(cycle+.5)%1],axis=1)
    support=a['foot_force']>.1*mass*9.81
    speed=np.linalg.norm(a['sole_vel'][...,:2],axis=-1)
    mask=support&(ph>.10)&(ph<.55)&(np.arange(n)[:,None]>=warm)
    # Force-threshold flicker is not a new landing. Require a clear 60ms swing.
    touchdown=np.zeros_like(support);air_ticks=np.zeros(2,dtype=int);armed=np.zeros(2,dtype=bool)
    for i in range(n):
        air_ticks=np.where(a['sole_low'][i]>.004,air_ticks+1,0)
        armed|=air_ticks>=3
        landed=armed&support[i]&(a['sole_low'][i]<.003)
        touchdown[i]=landed&(i>=warm)
        armed&=~landed
    peaks=[]
    for side in range(2):
        c=cycle+.5*side;values=[]
        for k in np.unique(np.floor(c)).astype(int):
            indices=np.flatnonzero((c>=k+.65)&(c<k+1)&(np.arange(n)>=warm))
            complete=(len(indices)>=3 and c[indices[0]]<=k+.68 and c[indices[-1]]>=k+.97 and c[indices[-1]]-c[indices[0]]>=.30) if variable else len(indices)>=int(.35*period/.02)-2
            if complete:values.append(float(a['sole_low'][indices,side].max()*1000))
        peaks.append(values)
    result=dict(settled_stance_speed_p95_m_s=float(np.percentile(speed[mask],95)) if mask.any() else None,
                touchdown_speed_p95_m_s=float(np.percentile(speed[touchdown],95)) if touchdown.any() else None,
                complete_swing_peak_clearances_mm=peaks,
                minimum_swing_peak_mm=[min(v) if v else None for v in peaks],
                median_swing_peak_mm=[float(np.median(v)) if v else None for v in peaks],
                mean_forward_speed_m_s=float(np.diff(a['xyz'][warm:,0]).mean()/.02) if n>warm+1 else None)
    result.update(max_lateral_from_initial_m=float(np.abs(a['xyz'][:,1]-a['xyz'][0,1]).max()),
                  final_lateral_displacement_m=float(a['xyz'][-1,1]-a['xyz'][0,1]))
    if force_limits is not None and actuator_names is not None:
        lim=np.asarray(force_limits);torque=np.abs(a['torque'][warm:]);valid=lim>0
        ratio=torque[:,valid]/lim[valid]
        result['actuator_load']={name:dict(peak_limit_ratio=float(ratio[:,i].max()),fraction_above_95pct_limit=float((ratio[:,i]>.95).mean()))
                                for i,name in enumerate(np.asarray(actuator_names)[valid])}
    return result
