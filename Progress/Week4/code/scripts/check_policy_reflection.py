"""Check reflection contracts using actual environment observations on GPU."""
import bootstrap
bootstrap.configure_warp()
from bootstrap import WEEK
import json,torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict
from sunday_locomotion.env_cfg import make_env
from sunday_locomotion.symmetry import mapping,reflect_group,reflect_actions,augment

torch.set_num_threads(4)
env=ManagerBasedRlEnv(make_env(32,'control',True,play=True,revision=16),device='cuda:0')
try:
    wrapper=RslRlVecEnvWrapper(env,clip_actions=1.)
    with torch.inference_mode():
        obs,_=env.reset();term=env.command_manager.get_term('twist');term.override=[.05,0,.12]
        for _ in range(75):obs,_,_,_,_=env.step(torch.zeros(32,22,device=env.device))
    m=mapping(wrapper);original=TensorDict(obs,batch_size=[32]);before=original.clone()
    values={}
    for group in original.keys():
        a=original[group];b=reflect_group(a,group,m);c=reflect_group(b,group,m)
        values[group+'_double_reflection_max_error']=float((a-c).abs().max())
        assert values[group+'_double_reflection_max_error']<1e-6
    actions=torch.rand(32,22,device=env.device)*1.6-.8
    values['action_double_reflection_max_error']=float((actions-reflect_actions(reflect_actions(actions,m),m)).abs().max())
    assert values['action_double_reflection_max_error']<1e-6
    expanded,reflected=augment(wrapper,original,actions)
    assert expanded.batch_size==torch.Size([64]) and reflected.shape==(64,22)
    assert all(torch.equal(original[k],before[k]) for k in original.keys())
    assert torch.allclose(expanded['actor'][32:,44],-original['actor'][:,44])
    assert torch.allclose(expanded['critic'][32:,92],original['critic'][:,93])
    values.update(passed=True,actor_dim=460,critic_dim=98,actions=22,observations=32,augmented_observations=64,uses_actual_CAD_axes=True,uses_nominal_joint_offsets=True,data_augmentation_for_PPO_likelihood_allowed=False)
    (WEEK/'results/validation/policy_reflection.json').write_text(json.dumps(values,indent=2),encoding='utf-8');print(json.dumps(values),flush=True)
finally:env.close()
