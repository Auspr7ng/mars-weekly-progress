"""Rescore saved joystick traces over whole gait cycles; preserve raw verdicts."""
import bootstrap
from bootstrap import WEEK
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sunday_locomotion.cadence import native_period


def average(t,values,start,end):
    def at(x):
        i=int(np.searchsorted(t,x))
        if i<len(t) and abs(t[i]-x)<1e-7:return values[i]
        if i==0 or i==len(t):raise ValueError('Mean window extends beyond recorded state samples')
        f=(x-t[i-1])/(t[i]-t[i-1]);return values[i-1]*(1-f)+values[i]*f
    mask=(t>start)&(t<end)
    times=np.concatenate(([start],t[mask],[end]));v=np.concatenate((at(start)[None],values[mask],at(end)[None]),axis=0)
    dt=np.diff(times).reshape((-1,)+(1,)*(v.ndim-1))
    return (((v[:-1]+v[1:])*.5)*dt).sum(axis=0)/(end-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--native-result',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();original=json.loads(args.evaluation.read_text(encoding='utf-8'))
    source=original;native=bool(args.native_result)
    if native:
        source=json.loads(args.native_result.read_text(encoding='utf-8'))
        assert source['joystick_limits']==original['limits']
        a=dict(np.load(args.native_result.with_name(args.native_result.stem+'_web_joystick_trace.npz')))
        a['low']=a.pop('sole_low');a['force']=a.pop('foot_force');survived=source['programs'][0]['successes']
    else:a=dict(np.load(args.evaluation.with_suffix('.npz')));survived=original['survived']
    old_hash=hashlib.sha256(Path(original['checkpoint']).read_bytes()).hexdigest();new_hash=hashlib.sha256(Path(source['checkpoint']).read_bytes()).hexdigest();assert old_hash==new_hash
    spec=json.loads((Path(source['checkpoint']).parent/'run.json').read_text(encoding='utf-8'))
    mass=json.loads((Path(source['checkpoint']).parent/'dynamics/arm_home.json').read_text(encoding='utf-8'))['total_mass_kg']
    t=a['time'];n=a['velocity'].shape[1];passed=a['alive'].all(axis=0).copy();rows=[]
    assert int(a['alive'][-1].sum())==survived,'Missing final survival mask: rerun with complete final-state records'
    ticks=np.zeros((n,2),int);armed=np.zeros((n,2),bool);landings=[]
    for low,force,live in zip(a['low'],a['force'],a['alive']):
        ticks=np.where(low>.004,ticks+1,0);armed|=ticks>=3
        landing=armed&(force>.1*mass*9.81)&(low<.003)&live[:,None];armed&=~landing;landings.append(landing)
    landings=np.asarray(landings)
    for old in original['stages']:
        start=old['start_s'];end=old['end_s'];short=end-start<1.21;begin=start+(.6 if short else 1.2)
        ix=np.flatnonzero((t>=begin)&(t<end));ok=a['alive'][ix].all(axis=0)
        row=dict(stage=old['stage'],start_s=start,end_s=end,held=old['held'],original_passed=old['passed'] if not native else None)
        if not old['held']:
            speed=np.percentile(np.linalg.norm(a['velocity'][ix,:,:2],axis=-1),95,axis=0)
            yaw=np.percentile(np.abs(a['omega'][ix,:,2]),95,axis=0);drift=np.linalg.norm(a['xyz'][ix[-1],:,:2]-a['xyz'][ix[0],:,:2],axis=-1)
            both=(a['force'][ix]>.1*mass*9.81).all(axis=-1).mean(axis=0)
            ok&=(speed<.025)&(yaw<.10)&(drift<.04)&(both>.85)
        else:
            command=np.asarray(old['command']);period=float(native_period(command)) if spec['revision']>=25 else 1.4
            cycles=int(np.floor((end-begin)/period+1e-7)) if not short else 0
            if cycles:
                mean_start=end-cycles*period;vx=average(t,a['velocity'][:,:,0],mean_start,end);wz=average(t,a['omega'][:,:,2],mean_start,end)
            else:mean_start=begin;vx=a['velocity'][ix,:,0].mean(axis=0);wz=a['omega'][ix,:,2].mean(axis=0)
            if abs(command[0])>.001:ok&=vx*np.sign(command[0])>=(.015 if short else (.12 if abs(command[0])>=.14 else .70*abs(command[0])))
            if abs(command[2])>.001:ok&=wz*np.sign(command[2])>=(.08 if short else .70*abs(command[2]))
            if not short:ok&=landings[(t>=start)&(t<end)].sum(axis=0).min(axis=1)>=1
            row.update(command=command.tolist(),whole_cycles=cycles,mean_window_start_s=mean_start,mean_window_end_s=end,
                cycle_forward_mean=float(vx.mean()),cycle_yaw_rate_mean=float(wz.mean()))
        passed&=ok;row['passed']=int(ok.sum());rows.append(row)
    out=dict(checkpoint=source['checkpoint'],checkpoint_sha256=old_hash,original_evaluation=str(args.native_result or args.evaluation),original_accepted=original['accepted'] if not native else source['all_requested_programs_passed'],
        episodes=n,survived=survived,passed=int(passed.sum()),accepted=bool(original['release_zero_contract'] and survived/n>=.95 and passed.mean()>=.90),
        raw_release_contract=original['release_zero_contract'],limits=original['limits'],stages=rows,
        scope='Same recorded physics; non-short held velocity is averaged over whole gait cycles within the existing steady window. Original70% thresholds, release quietness, support/landing and short-burst checks remain unchanged.')
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
