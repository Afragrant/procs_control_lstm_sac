"""采集和存盘沿用 RecurrentSACAgent，学习器换成序列上的条件 CEM。"""

from __future__ import annotations

from algorithm.Recurrent_SAC.agent import RecurrentSACAgent

from .learner import RecurrentGreedySACLearner


class RecurrentGreedySACAgent(RecurrentSACAgent):
    def _make_learner(self):
        return RecurrentGreedySACLearner(
            self.config, self.obs_dim, self.action_dim, self.device, self.obs_norm
        )
