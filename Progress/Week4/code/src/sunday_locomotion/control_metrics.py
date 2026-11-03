"""Command-following checks from first-episode trajectories, independent of reward."""
import numpy as np
from .command_programs import segments


def summarize(a,program,mass):
    n=a['xyz'].shape[1];result=[];case_pass=np.ones(n,dtype=bool)
    support=a['foot_force']>.1*mass*9.81
    # Confirm a 60ms whole-sole swing, then one supported landing.
    landing=np.zeros_like(support);ticks=np.zeros((n,2),dtype=int);armed=np.zeros((n,2),dtype=bool)
    for i in range(len(a['time'])):
        ticks=np.where(a['sole_low'][i]>.004,ticks+1,0);armed|=ticks>=3
        event=armed&support[i]&(a['sole_low'][i]<.003)&a['alive'][i,:,None]
        landing[i]=event;armed&=~event
    for segment in segments(program):
        start,end=segment['start'],segment['end'];kind=segment['kind']
        full=(a['time']>=start)&(a['time']<end)
        steady=full&(a['time']>=start+1.2)
        ix=np.flatnonzero(steady)
        if not len(ix):continue
        alive=a['alive'][ix].all(axis=0)
        v=a['velocity'][ix];cmd=a['command'][ix];omega=a['omega'][ix,:,2]
        forward_rmse=np.sqrt(np.mean((v[:,:,0]-cmd[:,:,0])**2,axis=0))
        yaw_rmse=np.sqrt(np.mean((omega-cmd[:,:,2])**2,axis=0))
        heading=np.sqrt(np.mean(a['heading_error'][ix]**2,axis=0))
        planar=np.linalg.norm(v[:,:,:2],axis=-1)
        counts=landing[full].sum(axis=0)
        passed=alive&(heading<.25)
        drift=np.linalg.norm(a['xyz'][ix[-1],:,:2]-a['xyz'][ix[0],:,:2],axis=-1)
        row=dict(**segment,forward_rmse_mean=float(forward_rmse.mean()),yaw_rate_rmse_mean=float(yaw_rmse.mean()),
                 mean_forward_velocity=float(v[:,:,0].mean()),mean_yaw_rate=float(omega.mean()),heading_rmse_mean=float(heading.mean()),
                 mean_landings_left_right=counts.mean(axis=0).tolist())
        if kind=='stop':
            speed95=np.percentile(planar,95,axis=0);yaw95=np.percentile(np.abs(omega),95,axis=0)
            both=support[ix].all(axis=-1).mean(axis=0)
            passed&=(speed95<.025)&(yaw95<.10)&(drift<.04)&(both>.85)
            # First sustained 0.5s quiet double support; report missing settling as null.
            quiet=(np.linalg.norm(a['velocity'][full,:,:2],axis=-1)<.025)&(np.abs(a['omega'][full,:,2])<.10)&support[full].all(axis=-1)&a['alive'][full]
            streak=np.zeros(n,dtype=int);latency=np.full(n,np.nan)
            times=a['time'][full]
            for j in range(len(times)):
                streak=np.where(quiet[j],streak+1,0);new=(streak>=25)&np.isnan(latency);latency[new]=times[j]-start
            passed&=np.isfinite(latency)&(latency<=1.8)
            row.update(planar_speed_p95_mean=float(speed95.mean()),abs_yaw_rate_p95_mean=float(yaw95.mean()),
                       drift_after_settling_mean_m=float(drift.mean()),double_support_fraction_mean=float(both.mean()),
                       settled_fraction=float(np.isfinite(latency).mean()),settling_latency_mean_s=float(np.nanmean(latency)) if np.isfinite(latency).any() else None)
        else:
            passed&=(forward_rmse<.04)&(yaw_rmse<.08)&(counts.min(axis=1)>=1)
            target_v=segment['command'][0];target_w=segment['command'][2]
            if abs(target_v)>=.14:passed&=v[:,:,0].mean(axis=0)*np.sign(target_v)>=.12
            elif abs(target_v)>=.11:passed&=v[:,:,0].mean(axis=0)*np.sign(target_v)>=.70*abs(target_v)
            if target_v<-.04:passed&=-v[:,:,0].mean(axis=0)>=.70*abs(target_v)
            if abs(target_w)>=.35:passed&=omega.mean(axis=0)*np.sign(target_w)>=.28
            if kind=='in_place':passed&=drift<.08
            if segment['command'][2]:
                j=np.flatnonzero(full);actual=a['yaw_unwrapped'][j[-1]]-a['yaw_unwrapped'][j[0]]
                expected=a['command'][j,:,2].sum(axis=0)*.02
                passed&=(actual*expected>0)&(np.abs(actual)>.6*np.abs(expected))
                row.update(mean_yaw_change_rad=float(actual.mean()),mean_command_yaw_integral_rad=float(expected.mean()))
        case_pass&=passed;row.update(passed_episodes=int(passed.sum()),pass_rate=float(passed.mean()))
        result.append(row)
    survived=a['alive'][-1];case_pass&=survived
    # Straight speed changes must not hide accumulated lateral drift.
    if program in ('steady','speed_stop','fast_steady','fast_speed_stop','reverse_stop','responsive_steady','responsive_speed_stop','responsive_reverse','responsive_backstep'):
        max_lateral=np.abs(a['xyz'][:,:,1]-a['xyz'][0,:,1]).max(axis=0)
        case_pass&=max_lateral<(.15 if program in ('steady','fast_steady','responsive_steady') else .20)
    else:max_lateral=None
    return dict(program=program,episodes=n,seconds=float(a['time'][-1]+.02),successes=int(survived.sum()),
                capability_passed_episodes=int(case_pass.sum()),capability_pass_rate=float(case_pass.mean()),
                engineering_passed=bool(survived.mean()>=.95 and case_pass.mean()>=.90),segments=result,
                max_lateral_m=float(max_lateral.max()) if max_lateral is not None else None,
                notes='First episode only. Stop checks begin 1.2s after the raw zero command; require sustained quiet double support. Turn sign and yaw amount are checked. These checks do not replace gait-quality/video review.')
