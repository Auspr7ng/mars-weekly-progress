"""Soft left/right policy consistency with actual CAD axes and nominal offsets.

The modified robot has sub-millimetre geometric asymmetry. This callback is for
a mirror loss, not for substituting reflected samples into the PPO likelihood.
"""
import hashlib,json
from pathlib import Path
import torch

WEEK=Path(__file__).resolve().parents[3]


def mapping(env):
    base=env.unwrapped
    if hasattr(base,'_sunday_mirror_mapping'):return base._sunday_mirror_mapping
    p=WEEK/'config/mirror_map.json';cfg=json.loads(p.read_text())
    if not cfg['approximate_prior_usable']:raise ValueError('Reflection prior was not validated')
    if hashlib.sha256((WEEK/'model/sunday_a1_locomotion.xml').read_bytes()).hexdigest()!=cfg['model_sha256']:raise ValueError('Reflection geometry has changed')
    robot=base.scene['robot'];names=list(robot.joint_names);action=base.action_manager.get_term('residual')
    geometry_names=cfg['joint_names'];mates=[];signs=[]
    for n in names:
        i=geometry_names.index(n);mates.append(names.index(geometry_names[cfg['mirror_indices'][i]]));signs.append(cfg['mirror_signs'][i])
    device=base.device;index=torch.tensor(mates,device=device);sign=torch.tensor(signs,device=device)
    nominal=robot.data.default_joint_pos[0].clone();offset=sign*nominal[index]-nominal
    action_names=list(action.names);amates=[action_names.index(names[mates[names.index(n)]]) for n in action_names]
    ai=torch.tensor(amates,device=device);aids=action.ids;scale=action.scales
    asign=sign[aids]*scale[ai]/scale;aoffset=offset[aids]/scale
    result=dict(index=index,sign=sign,offset=offset,action_index=ai,action_sign=asign,action_offset=aoffset)
    base._sunday_mirror_mapping=result
    return result


def reflect_actions(actions,m):
    return actions[...,m['action_index']]*m['action_sign']+m['action_offset']


def reflect_group(x,group,m):
    out=x.clone();lead=x.shape[:-1]
    def vector(start,hist,sign):
        out[...,start:start+hist*3]=(x[...,start:start+hist*3].reshape(*lead,hist,3)*torch.tensor(sign,device=x.device)).flatten(-2)
    def joints(start,hist,velocity=False):
        q=x[...,start:start+hist*27].reshape(*lead,hist,27)
        y=q[...,m['index']]*m['sign']
        if not velocity:y=y+m['offset']
        out[...,start:start+hist*27]=y.flatten(-2)
    def actions(start,hist):
        q=x[...,start:start+hist*22].reshape(*lead,hist,22)
        out[...,start:start+hist*22]=reflect_actions(q,m).flatten(-2)
    if group=='actor':
        if x.shape[-1]!=460:raise ValueError('Unexpected actor reflection schema')
        vector(0,5,[-1,1,-1]);vector(15,5,[1,-1,1]);vector(30,5,[1,-1,-1])
        joints(45,5);joints(180,5,True);actions(315,5)
        out[...,425:435]=-x[...,425:435]
        out[...,435:445]=(x[...,435:445].reshape(*lead,5,2)*torch.tensor([-1,1],device=x.device)).flatten(-2)
        vector(445,5,[1,-1,1])
    elif group=='critic':
        if x.shape[-1]!=98:raise ValueError('Unexpected critic reflection schema')
        vector(0,1,[-1,1,-1]);vector(3,1,[1,-1,1]);vector(6,1,[1,-1,-1])
        joints(9,1);joints(36,1,True);actions(63,1)
        out[...,85:87]=-x[...,85:87];out[...,87:89]=x[...,87:89]*torch.tensor([-1,1],device=x.device)
        vector(89,1,[1,-1,1]);out[...,92:98]=x[..., [93,92,95,94,97,96]]
    else:raise ValueError(f'Unknown reflection observation group: {group}')
    return out


def augment(env,obs=None,actions=None):
    m=mapping(env)
    if obs is not None:
        mirrored=obs.clone()
        for group in obs.keys():mirrored[group]=reflect_group(obs[group],group,m)
        obs=torch.cat([obs,mirrored],dim=0)
    if actions is not None:actions=torch.cat([actions,reflect_actions(actions,m)],dim=0)
    return obs,actions
