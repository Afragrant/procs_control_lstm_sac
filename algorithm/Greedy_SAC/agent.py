"""采集和存盘沿用 SACAgent，学习器换成条件交叉熵。"""

from __future__ import annotations

from algorithm.SAC.agent import SACAgent

from .learner import GreedySACLearner


class GreedySACAgent(SACAgent):
    def _make_learner(self):
        return GreedySACLearner(
            self.config, self.obs_shape, self.action_dim, self.device, obs_norm=self.obs_norm
        )
