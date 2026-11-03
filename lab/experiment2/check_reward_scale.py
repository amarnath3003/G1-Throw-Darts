"""Verify critic-unit conversion, actor preservation, serialization and wrapper."""
import runtime_bootstrap
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from train import RewardScale,rescale_value_estimator


def main():
    torch.set_num_threads(1)
    exp=Path(__file__).resolve().parent
    model=PPO.load(str(exp/'models/best_verified_20261031/policy.zip'),device='cpu')
    obs=np.random.default_rng(1901).normal(size=(64,73)).astype(np.float32)
    tensor=torch.as_tensor(obs)
    actor_before={k:v.clone() for k,v in model.policy.state_dict().items() if not k.startswith('value_net.')}
    actions_before=model.predict(obs,deterministic=True)[0]
    with torch.no_grad():values_before=model.policy.predict_values(tensor).clone()
    rescale_value_estimator(model,.02)
    for k,v in actor_before.items():assert torch.equal(v,model.policy.state_dict()[k]),k
    np.testing.assert_array_equal(actions_before,model.predict(obs,deterministic=True)[0])
    with torch.no_grad():torch.testing.assert_close(model.policy.predict_values(tensor),values_before*.02,atol=1e-6,rtol=1e-5)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'policy.zip';model.save(path)
        loaded=PPO.load(path,device='cpu')
        np.testing.assert_array_equal(actions_before,loaded.predict(obs,deterministic=True)[0])
        with torch.no_grad():torch.testing.assert_close(loaded.policy.predict_values(tensor),values_before*.02,atol=1e-6,rtol=1e-5)
    class Fixture(gym.Env):
        observation_space=gym.spaces.Box(-1.,1.,shape=(1,),dtype=np.float32)
        action_space=gym.spaces.Box(-1.,1.,shape=(1,),dtype=np.float32)
        def reset(self,**kwargs):return np.zeros(1,dtype=np.float32),{}
        def step(self,action):return np.zeros(1,dtype=np.float32),100.,True,False,{'physical_reward':100.}
    wrapper=RewardScale(Fixture(),.02)
    result=wrapper.step(np.zeros(1));assert result[1]==2. and result[4]['physical_reward']==100.
    output=exp/'analysis/ppo_review_20261031/reward_scale_checks.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(passed=True,scale=.02,actor_actions_unchanged=True,
        critic_values_scaled=True,checkpoint_roundtrip=True,physical_info_preserved=True,
        note='Numerical regression tests; no performance improvement is claimed.'),indent=2))
    print('Reward-scale checks passed',flush=True)


if __name__=='__main__':main()
