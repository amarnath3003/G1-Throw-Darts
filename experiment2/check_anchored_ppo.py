"""CUDA gradient, update, frozen-reference and serialization regression checks."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from anchored_ppo import AnchoredPPO
from stable_baselines3 import PPO
class Fixture(gym.Env):
    observation_space=gym.spaces.Box(-np.inf,np.inf,(73,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(10,),dtype=np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);return np.zeros(73,dtype=np.float32),{}
    def step(self,a):return np.zeros(73,dtype=np.float32),float(-np.square(a).mean()),True,False,{}
source=Path(__file__).parent/'runs/20261101T085001250725/best_physics/policy.zip'
p=AnchoredPPO.load(source,device='cuda',env=DummyVecEnv([Fixture]),n_steps=8,batch_size=8,n_epochs=1)
p.set_anchor(source,1.)
obs=torch.zeros((8,73),device='cuda');assert abs(p.anchor_divergence(obs).item())<1e-7
anchor={k:v.clone() for k,v in p.anchor_policy.state_dict().items()}
with torch.no_grad():p.policy.action_net.bias[0]+=.01
kl=p.anchor_divergence(obs);assert kl.item()>0;kl.backward()
assert torch.isfinite(p.policy.action_net.bias.grad).all()
assert all(v.grad is None for v in p.anchor_policy.parameters())
p.policy.optimizer.zero_grad();p.learn(8)
assert all(torch.equal(v,p.anchor_policy.state_dict()[k]) for k,v in anchor.items())
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'policy';p.save(path);loaded=PPO.load(path,device='cuda')
 a=p.predict(np.zeros((1,73),dtype=np.float32),deterministic=True)[0]
 np.testing.assert_array_equal(a,loaded.predict(np.zeros((1,73),dtype=np.float32),deterministic=True)[0])
print('PASS: CUDA PPO update, finite anchor gradient, unchanged reference, ordinary evaluation reload.')
