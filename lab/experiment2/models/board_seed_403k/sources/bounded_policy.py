"""PPO policy using SB3's tanh-Gaussian with its exact log-prob correction.

Parameters retain standard MlpPolicy names/shapes for explicit warm migration.
The environment action space is symmetric [-1,1], so no additional rescale is
needed. This is a distribution change, not post-hoc clipping of Gaussian samples.
"""
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


class BoundedActorCriticPolicy(ActorCriticPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        if self.use_sde:raise ValueError('BoundedActorCriticPolicy uses diagonal Gaussian, not gSDE')
        if not (np.all(self.action_space.low==-1) and np.all(self.action_space.high==1)):
            raise ValueError('This policy requires a symmetric [-1,1] action space')
        # The superclass built the identical mean head/log_std and optimizer.
        # Its distribution dispatch accepts this DiagGaussian subclass and PPO
        # uses -log_prob when analytical entropy is unavailable.
        self.action_dist=SquashedDiagGaussianDistribution(self.action_space.shape[0])
