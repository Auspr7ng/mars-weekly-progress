"""Dense hand/leg clearance feedback using the existing collision boxes."""
import torch


def box_separation(ca,ra,ha,cb,rb,hb):
    """Signed SAT separation in metres; positive iff these oriented boxes separate."""
    aa=ra.transpose(-1,-2);ab=rb.transpose(-1,-2)
    crosses=torch.linalg.cross(aa.unsqueeze(-2),ab.unsqueeze(-3),dim=-1).flatten(-3,-2)
    axes=torch.cat((aa,ab,crosses),dim=-2)
    lengths=axes.norm(dim=-1,keepdim=True)
    valid=lengths[...,0]>1e-6
    axes=axes/lengths.clamp(min=1e-6)
    projected=((cb-ca).unsqueeze(-2)*axes).sum(dim=-1).abs()
    ar=(torch.matmul(axes,ra).abs()*ha.unsqueeze(-2)).sum(dim=-1)
    br=(torch.matmul(axes,rb).abs()*hb.unsqueeze(-2)).sum(dim=-1)
    return (projected-ar-br).masked_fill(~valid,-1e6).amax(dim=-1)


def hand_leg_separation(env):
    if not hasattr(env,'_hand_leg_box_pairs'):
        model=env.sim.mj_model;pairs=[]
        for side in ('l','r'):
            for hand in ('gripper_base','finger_top','finger_bottom'):
                for leg in ('hip_link','thigh_link','knee_link','calf_link'):
                    pairs.append((model.geom(f'robot/collision_{side}_{hand}_1').id,
                                  model.geom(f'robot/collision_{side}_{leg}_1').id))
        indices=torch.tensor(pairs,device=env.device)
        half=torch.as_tensor(model.geom_size.copy(),device=env.device,dtype=torch.float32)
        env._hand_leg_box_pairs=(indices[:,0],indices[:,1],half[indices[:,0]],half[indices[:,1]])
    a,b,ha,hb=env._hand_leg_box_pairs
    pos=env.sim.data.geom_xpos
    rotation=env.sim.data.geom_xmat.reshape(env.num_envs,-1,3,3)
    separation=box_separation(pos[:,a],rotation[:,a],ha,pos[:,b],rotation[:,b],hb)
    # Keep the closest hand/leg pair on each side; average sides, never dilute
    # an imminent collision by averaging with many distant geometry pairs.
    return separation.reshape(env.num_envs,2,-1).amin(dim=-1)


def hand_clearance_cost(env):
    separation=hand_leg_separation(env)
    margin=.018 if env.cfg.sunday_revision>=23 else (.012 if env.cfg.sunday_revision>=21 else .008)
    return ((margin-separation)/margin).clamp(min=0,max=3).square().mean(dim=-1)


def foot_clearance_cost(env):
    if not hasattr(env,'_foot_box_pair'):
        m=env.sim.mj_model
        a=m.geom('robot/collision_l_foot_link_1').id;b=m.geom('robot/collision_r_foot_link_1').id
        env._foot_box_pair=(a,b,torch.tensor(m.geom_size[a].copy(),device=env.device,dtype=torch.float32),
                           torch.tensor(m.geom_size[b].copy(),device=env.device,dtype=torch.float32))
    a,b,ha,hb=env._foot_box_pair;p=env.sim.data.geom_xpos;r=env.sim.data.geom_xmat.reshape(env.num_envs,-1,3,3)
    gap=box_separation(p[:,a],r[:,a],ha,p[:,b],r[:,b],hb)
    margin=.018 if env.cfg.sunday_revision>=25 else .008
    return ((margin-gap)/margin).clamp(min=0,max=3).square()


def torso_oscillation_cost(env):
    from mjlab.utils.lab_api.math import matrix_from_quat
    d=env.scene['robot'].data
    r=matrix_from_quat(d.root_link_quat_w)
    roll=torch.atan2(r[:,2,1],r[:,2,2])
    ground=env.scene.env_origins[:,2]
    height_error=(d.root_link_pos_w[:,2]-ground-env.cfg.sunday_h0).abs()
    roll_margin=.052359878 if env.cfg.sunday_revision>=21 else .061086524
    roll_cost=((roll.abs()-roll_margin)/.025).clamp(min=0).square()
    height_cost=((height_error-.004)/.004).clamp(min=0).square()
    return (roll_cost+height_cost).clamp(max=10)
