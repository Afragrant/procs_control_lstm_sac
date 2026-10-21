"""DDPG：确定性 actor、单 Q、目标网络软更新。观测是当前帧，没有历史。"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.objectives import DDPGLoss, SoftUpdate

from algorithm.SAC.policy import (
    OBS_NORMED_KEY,
    RunningObsNorm,
    build_deterministic_actor,
    build_qvalue,
)

LOSS_FUNCTION = 'l2'


class DDPGLearner:
    def __init__(
        self,
        config,
        obs_shape: tuple[int, ...],
        action_dim: int,
        device: torch.device,
        obs_norm: RunningObsNorm | None = None,
    ):
        self.config = config
        self.device = device
        self.obs_norm = obs_norm
        self.use_grad_clip = bool(getattr(config, 'use_grad_clip', False))
        self.grad_clip_norm = float(getattr(config, 'grad_clip_norm', 0.5))

        self.actor = build_deterministic_actor(config.representation, obs_shape, action_dim, config, device)
        self.loss_module = DDPGLoss(
            actor_network=self.actor,
            value_network=build_qvalue(config.representation, obs_shape, action_dim, config, device),
            loss_function=LOSS_FUNCTION,
            delay_actor=True,
            delay_value=True,
        )
        self.loss_module.make_value_estimator(gamma=float(config.gamma))
        self.target_updater = SoftUpdate(self.loss_module, tau=float(config.tau))

        self.optimizer = {
            'actor': torch.optim.Adam(
                self.loss_module.actor_network_params.flatten_keys().values(),
                lr=float(config.learning_rate_actor),
            ),
            'critic': torch.optim.Adam(
                self.loss_module.value_network_params.flatten_keys().values(),
                lr=float(config.learning_rate_critic),
            ),
        }

    def _normalize_batch(self, batch: TensorDict) -> TensorDict:
        if self.obs_norm is None:
            return batch
        batch[OBS_NORMED_KEY] = self.obs_norm.normalize(batch['observation'])
        batch['next', OBS_NORMED_KEY] = self.obs_norm.normalize(batch['next', 'observation'])
        return batch

    def update(self, batch) -> dict[str, float]:
        batch = self._normalize_batch(batch.to(self.device))
        out = self.loss_module(batch)

        self.optimizer['actor'].zero_grad()
        out['loss_actor'].backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.loss_module.actor_network_params.flatten_keys().values(), self.grad_clip_norm)
        self.optimizer['actor'].step()

        self.optimizer['critic'].zero_grad()
        out['loss_value'].backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(
                self.loss_module.value_network_params.flatten_keys().values(), self.grad_clip_norm
            )
        self.optimizer['critic'].step()

        self.target_updater.step()
        return {
            'Qloss': float(out['loss_value'].detach()),
            'Ploss': float(out['loss_actor'].detach()),
        }

    def state_dict(self) -> dict:
        return {
            'loss_module': self.loss_module.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizer.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.loss_module.load_state_dict(state['loss_module'], strict=True)
        for name, opt_state in state.get('optimizers', {}).items():
            if name in self.optimizer:
                self.optimizer[name].load_state_dict(opt_state)
