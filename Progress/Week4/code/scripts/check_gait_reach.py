"""Check the complete foot-space reward schedule against unchanged joint/action limits."""
import bootstrap
from bootstrap import WEEK
import argparse,json,math
import mujoco,numpy as np
from scipy.optimize import least_squares

p=argparse.ArgumentParser();p.add_argument('--speed',type=float,default=.1);p.add_argument('--pose',default='arm_home');p.add_argument('--height',type=float);p.add_argument('--period',type=float,default=1.4);p.add_argument('--ankle-scale',type=float,default=.45);p.add_argument('--sway',type=float,default=.028);args=p.parse_args()
h=json.loads((WEEK/f'model/{args.pose}.json').read_text());m=mujoco.MjModel.from_xml_path(str(WEEK/'model/scene.xml'));d=mujoco.MjData(m)
names=h['leg_order'];adr=[m.joint(n).qposadr[0] for n in names];q0=np.array([h['joint_pos'][n] for n in names])
lo=np.array([m.jnt_range[m.joint(n).id,0] for n in names]);hi=np.array([m.jnt_range[m.joint(n).id,1] for n in names]);mid=(lo+hi)/2
scale=np.array([args.ankle_scale if n.endswith('ankle_roll') else (.65 if 'knee' in n else (.45 if 'roll' in n else .35)) for n in names]);lo=np.maximum(mid+.9*(lo-mid),q0-scale);hi=np.minimum(mid+.9*(hi-mid),q0+scale)
sites=[m.site(s+'_sole').id for s in ('l','r')]
def assign(q,y):
    d.qpos[:]=m.qpos0;d.qpos[:3]=[0,y,h['root_height'] if args.height is None else args.height]
    for n,v in h['joint_pos'].items():d.qpos[m.joint(n).qposadr[0]]=v
    d.qpos[adr]=q;mujoco.mj_forward(m,d)
assign(q0,0);base=d.site_xpos[sites].copy();rot=d.site_xmat[sites].copy();reports=[];last=q0
for phase in np.linspace(0,1,80,endpoint=False):
    ph=np.array([phase,(phase+.5)%1]);stance=ph<.65;u=np.clip((ph-.65)/.35,0,1)
    span=args.speed*args.period*.65;backward=-args.speed*args.period*.35
    swing=(2*u**3-3*u**2+1)*(-span/2)+(u**3-2*u**2+u)*backward+(-2*u**3+3*u**2)*(span/2)+(u**3-u**2)*backward
    x=np.where(stance,span/2-args.speed*args.period*ph,swing);z=.020*np.sin(np.pi*u)**2*(~stance);y=args.sway*np.cos(2*np.pi*(phase-.325))
    target=base.copy();target[:,0]+=x;target[:,2]=.0004+z
    def error(q):
        assign(q,y)
        return np.r_[(d.site_xpos[sites]-target).ravel()*200,(d.site_xmat[sites]-rot).ravel()*12,(q-q0)*.01]
    candidates=[least_squares(error,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=300) for seed in (last,q0)]
    sol=min(candidates,key=lambda c:np.sum(error(c.x)**2));last=sol.x;assign(sol.x,y)
    rotation=d.site_xmat[sites].reshape(2,3,3);lowest=d.site_xpos[sites,2]-np.abs(rotation[:,2,0])*.0525-np.abs(rotation[:,2,1])*.09
    reports.append(dict(phase=float(phase),max_position_error_mm=float(np.linalg.norm(d.site_xpos[sites]-target,axis=1).max()*1000),
                        sole_clearance_mm=(lowest*1000).tolist(),target_clearance_mm=(z*1000).tolist(),q=sol.x.tolist()))
report=dict(speed_m_s=args.speed,period_s=args.period,sway_m=args.sway,ankle_action_scale=args.ankle_scale,pose=args.pose,root_height=h['root_height'] if args.height is None else args.height,max_position_error_mm=max(r['max_position_error_mm'] for r in reports),samples=reports,
            notes='Kinematic diagnostic only; no root driving or reference trajectory is used in PPO.')
suffix='' if args.height is None else f'_z{args.height:.3f}'
if args.period!=1.4:suffix+=f'_T{args.period:.2f}'
if args.ankle_scale!=.45:suffix+=f'_a{args.ankle_scale:.2f}'
if args.sway!=.028:suffix+=f'_sway{args.sway:.3f}'
(WEEK/f'results/quality/cycle_reach_{args.pose}_{args.speed:.2f}{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
