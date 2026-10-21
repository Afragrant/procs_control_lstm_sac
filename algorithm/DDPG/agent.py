"""采集和存盘沿用 SACAgent。策略是确定性的，采集时在动作上加高斯噪声。"""

from __future__ import annotations

import math

from tensordict.nn import TensorDictSequential
from torchrl.data import Bounded
from torchrl.modules import AdditiveGaussianModule

from algorithm.SAC.agent import SACAgent
from algorithm.SAC.policy import build_obs_norm_layer

from .learner import DDPGLearner


class DDPGAgent(SACAgent):
    def _make_learner(self):
        return DDPGLearner(self.config, self.obs_shape, self.action_dim, self.device, obs_norm=self.obs_norm)

    def __init__(self, config, envs=None):
        super().__init__(config, envs)
        noise = float(getattr(config, 'exploration_noise', 0.1))
        if not math.isfinite(noise) or noise < 0:
            raise ValueError(f'exploration_noise must be finite and non-negative, got {noise}')
        if noise == 0.0:
            return
        # shortcut: 白噪声，动作变化惩罚开始主导回报时再换成 OU，并给环境加 InitTracker。
        spec = Bounded(-1.0, 1.0, (self.action_dim,), device=self.device)
        self.collector_policy = TensorDictSequential(
            build_obs_norm_layer(self.obs_norm),
            self.policy,
            AdditiveGaussianModule(spec=spec, sigma_init=1.0, sigma_end=1.0, std=noise, device=self.device),
        )
