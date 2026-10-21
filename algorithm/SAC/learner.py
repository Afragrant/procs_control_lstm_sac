"""SAC learner built on torchrl's SACLoss (double Q, target networks, alpha tuning)."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.objectives import SACLoss, SoftUpdate

from .policy import OBS_NORMED_KEY, RunningObsNorm, build_actor, build_qvalue

LOSS_FUNCTION = 'l2'  # torchrl 的 l2 即 F.mse_loss


def resolve_target_entropy(config, action_dim: int) -> float:
    """配置里写了 target_entropy 就用它，包括 0；没写则用 -action_dim。"""
    if not hasattr(config, 'target_entropy'):
        return -float(action_dim)
    return float(config.target_entropy)


class SACLearner:
    """包装 torchrl SACLoss：双 Q 学习、目标网络软更新、自动熵调节。

    网络不含任何归一化模块（SACLoss 克隆 Q 集成会冻结 buffer 副本）。obs_norm
    仅作为独立预处理持有：update() 在送入 SACLoss 前，用同一套当前统计量把
    s（observation）与 s'（next.observation）归一化到 OBS_NORMED_KEY，
    保证转移两端的输入尺度一致。
    """

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
        self.use_automatic_entropy_tuning = bool(config.use_automatic_entropy_tuning)
        self.use_grad_clip = bool(getattr(config, 'use_grad_clip', False))
        self.grad_clip_norm = float(getattr(config, 'grad_clip_norm', 0.5))
        self.deactivate_vmap = representation_is_recurrent(config.representation)

        self.actor = build_actor(config.representation, obs_shape, action_dim, config, device)
        self.loss_module = SACLoss(
            actor_network=self.actor,
            qvalue_network=build_qvalue(config.representation, obs_shape, action_dim, config, device),
            num_qvalue_nets=2,
            loss_function=LOSS_FUNCTION,
            alpha_init=float(config.alpha),
            fixed_alpha=not self.use_automatic_entropy_tuning,
            target_entropy=resolve_target_entropy(config, action_dim),
            delay_actor=False,
            delay_qvalue=True,
            deactivate_vmap=self.deactivate_vmap,
        )
        self.loss_module.make_value_estimator(gamma=float(config.gamma))
        self.target_updater = SoftUpdate(self.loss_module, tau=float(config.tau))

        self.optimizer = {
            'actor': torch.optim.Adam(
                self.loss_module.actor_network_params.flatten_keys().values(),
                lr=float(config.learning_rate_actor),
            ),
            'critic': torch.optim.Adam(
                self.loss_module.qvalue_network_params.flatten_keys().values(),
                lr=float(config.learning_rate_critic),
            ),
        }
        if self.use_automatic_entropy_tuning:
            alpha_lr = float(getattr(config, 'learning_rate_alpha', config.learning_rate_actor))
            self.optimizer['alpha'] = torch.optim.Adam([self.loss_module.log_alpha], lr=alpha_lr)

    def _normalize_batch(self, batch: TensorDict) -> TensorDict:
        """用当前统计量把 s 与 s' 归一化到 OBS_NORMED_KEY（同一套统计量，两端一致）。"""
        if self.obs_norm is None:
            return batch
        batch[OBS_NORMED_KEY] = self.obs_norm.normalize(batch['observation'])
        batch['next', OBS_NORMED_KEY] = self.obs_norm.normalize(batch['next', 'observation'])
        return batch

    def update(self, batch) -> dict[str, float]:
        """对一个从 replay buffer 采出的 TensorDict 做一次梯度更新。"""
        batch = self._normalize_batch(batch.to(self.device))
        out = self.loss_module(batch)

        self.optimizer['actor'].zero_grad()
        out['loss_actor'].backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.loss_module.actor_network_params.flatten_keys().values(), self.grad_clip_norm)
        self.optimizer['actor'].step()

        self.optimizer['critic'].zero_grad()
        out['loss_qvalue'].backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(
                self.loss_module.qvalue_network_params.flatten_keys().values(), self.grad_clip_norm
            )
        self.optimizer['critic'].step()

        if self.use_automatic_entropy_tuning:
            self.optimizer['alpha'].zero_grad()
            out['loss_alpha'].backward()
            self.optimizer['alpha'].step()

        self.target_updater.step()

        info = {
            'Qloss': float(out['loss_qvalue'].detach()),
            'Ploss': float(out['loss_actor'].detach()),
        }
        if self.use_automatic_entropy_tuning:
            info['alpha'] = float(out['alpha'].detach())
        return info

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


def representation_is_recurrent(representation: str) -> bool:
    return representation == 'History_LSTM'
