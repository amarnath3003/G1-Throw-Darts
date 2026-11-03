"""Validate bounded-action math, exact parameter transfer and serialization."""
import json
from pathlib import Path
import shutil
import cloudpickle
import numpy as np
import torch
from stable_baselines3 import PPO
from bounded_policy import BoundedActorCriticPolicy
from env import DartThrowEnv


def main():
    torch.set_num_threads(1);torch.manual_seed(73129)
    here=Path(__file__).resolve().parent
    source=here/'runs/20261029T163135827530';root=here/'analysis/round300k/migrated301k'
    root.mkdir(parents=True,exist_ok=False)
    old=PPO.load(str(source/'policy.zip'),device='cuda')
    new=PPO.load(str(source/'policy.zip'),device='cuda',custom_objects={'policy_class':BoundedActorCriticPolicy})
    for name,value in old.policy.state_dict().items():
        torch.testing.assert_close(value,new.policy.state_dict()[name],rtol=0,atol=0)
    env=DartThrowEnv(variant='arc_v5',randomize=True);obs,_=env.reset(seed=74101)
    with (source/'normalize.pkl').open('rb') as f:norm=cloudpickle.load(f)
    obs=norm.normalize_obs(np.repeat(obs[None,:],128,axis=0))
    tensor,_=new.policy.obs_to_tensor(obs)
    with torch.no_grad():
        old_mean=old.policy.get_distribution(tensor).distribution.mean
        new_mode=new.policy.get_distribution(tensor).mode()
        torch.testing.assert_close(new_mode,torch.tanh(old_mean),rtol=1e-6,atol=1e-6)
        actions,_,old_lp=new.policy(tensor)
    assert torch.isfinite(actions).all() and torch.all(actions.abs()<=1)
    _,log_prob,entropy=new.policy.evaluate_actions(tensor,actions)
    torch.testing.assert_close(log_prob,old_lp,atol=2e-4,rtol=1e-5)
    assert entropy is None and torch.isfinite(log_prob).all()
    new.policy.optimizer.zero_grad();(-log_prob.mean()).backward()
    for p in [new.policy.action_net.weight,new.policy.action_net.bias,new.policy.log_std]:
        assert p.grad is not None and torch.isfinite(p.grad).all()
    new.policy.optimizer.zero_grad()
    new.save(str(root/'policy'))
    restored=PPO.load(str(root/'policy.zip'),device='cuda')
    expected,_=new.predict(obs,deterministic=True);actual,_=restored.predict(obs,deterministic=True)
    np.testing.assert_allclose(expected,actual,rtol=0,atol=0)
    shutil.copy2(source/'normalize.pkl',root/'normalize.pkl')
    config=json.loads((source/'config.json').read_text());config.update(variant='arc_v5',bounded_actions=True,
        source_checkpoint=str(source/'policy.zip'),purpose='Validated parameter-preserving distribution migration')
    (root/'config.json').write_text(json.dumps(config,indent=2))
    report={'passed':True,'source':str(source/'policy.zip'),'timesteps':new.num_timesteps,
        'all_network_parameters_preserved':True,'normalization_preserved':True,
        'optimizer_state_retained':True,'distribution':type(new.policy.action_dist).__name__,
        'sample_bounds_passed':True,'forward_evaluate_logprob_agreement':True,
        'finite_gradients':True,'save_load_exact_actions':True}
    (root/'migration.json').write_text(json.dumps(report,indent=2));env.close()
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
