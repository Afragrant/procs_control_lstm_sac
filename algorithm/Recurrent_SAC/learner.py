"""从回合前缀重建各网络自己的隐状态，再计算带 mask 的 SAC 损失。"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from algorithm.SAC.learner import resolve_target_entropy
from algorithm.SAC.policy import RunningObsNorm

from .encoder import replay_prefix
from .networks import RecurrentActor, RecurrentQ, action_distribution
from .replay import SequenceBatch


def extend_obs(obs: Tensor, next_obs: Tensor) -> Tensor:
    """学习段每个时刻的观测，再接上最后一帧的真实下一观测。中间帧的下一观测就是下一行观测。"""
    return torch.cat([obs, next_obs[:, -1:]], dim=1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def features_from_history(encoder, prefix: Tensor, lengths: Tensor, sequence: Tensor, *, grad: bool) -> Tensor:
    """前缀只用来恢复起点状态，不参与本次反向传播。"""
    with torch.no_grad():
        state = replay_prefix(encoder, prefix, lengths)
        state = tuple(part.detach() for part in state)
    if not grad:
        with torch.no_grad():
            features, _ = encoder(sequence, state)
            return features
    features, _ = encoder(sequence, state)
    return features


def sample_action(loc: Tensor, scale: Tensor, *, rsample: bool) -> tuple[Tensor, Tensor]:
    dist = action_distribution(loc, scale)
    action = dist.rsample() if rsample else dist.sample()
    log_prob = dist.log_prob(action)
    if log_prob.shape != loc.shape[:-1]:
        log_prob = log_prob.reshape(loc.shape[:-1])
    return action, log_prob


@contextmanager
def frozen(modules):
    params = [param for module in modules for param in module.parameters()]
    previous = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, flag in zip(params, previous, strict=True):
            param.requires_grad_(flag)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers(), strict=True):
        target_buffer.copy_(source_buffer)


def _hidden_size(config) -> int:
    hidden_sizes = list(config.representation_hidden_size)
    if not hidden_sizes or int(hidden_sizes[-1]) <= 0:
        raise ValueError(f'representation_hidden_size must end with a positive size, got {hidden_sizes}')
    return int(hidden_sizes[-1])


class RecurrentSACLearner:
    def __init__(self, config, obs_dim: int, action_dim: int, device: torch.device, obs_norm: RunningObsNorm | None):
        self.device = device
        self.obs_norm = obs_norm
        self.gamma = float(config.gamma)
        self.tau = float(config.tau)
        self.auto_alpha = bool(config.use_automatic_entropy_tuning)
        self.target_entropy = resolve_target_entropy(config, action_dim)
        self.use_grad_clip = bool(getattr(config, 'use_grad_clip', False))
        self.grad_clip_norm = float(getattr(config, 'grad_clip_norm', 0.5))
        hidden = _hidden_size(config)
        actor_sizes = list(config.actor_hidden_size)
        critic_sizes = list(config.critic_hidden_size)
        activation = config.activation

        self.actor = RecurrentActor(obs_dim, action_dim, hidden, actor_sizes, activation, device)
        self.q1 = RecurrentQ(obs_dim, action_dim, hidden, critic_sizes, activation, device)
        self.q2 = RecurrentQ(obs_dim, action_dim, hidden, critic_sizes, activation, device)
        self.target_q1 = copy.deepcopy(self.q1).eval()
        self.target_q2 = copy.deepcopy(self.q2).eval()
        for target in (self.target_q1, self.target_q2):
            for param in target.parameters():
                param.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(math.log(float(config.alpha)), device=device))
        min_alpha = getattr(config, 'min_alpha', None)
        self.min_log_alpha = None if min_alpha is None else math.log(float(min_alpha))

        self.optimizer = {
            'actor': torch.optim.Adam(self.actor.parameters(), lr=float(config.learning_rate_actor)),
            'critic': torch.optim.Adam(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                lr=float(config.learning_rate_critic),
            ),
        }
        if self.auto_alpha:
            alpha_lr = float(getattr(config, 'learning_rate_alpha', config.learning_rate_actor))
            self.optimizer['alpha'] = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def alpha(self) -> Tensor:
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(min=self.min_log_alpha)
        return self.log_alpha.detach().exp()

    def update(self, batch: SequenceBatch) -> dict[str, float]:
        batch = self._normalize(batch.to(self.device))
        q_loss = self.critic_loss(batch)
        actor_loss, log_prob = self.actor_loss(batch)
        self._step(self.optimizer['actor'], actor_loss, self.actor.parameters())
        self._step(self.optimizer['critic'], q_loss, list(self.q1.parameters()) + list(self.q2.parameters()))
        info = {
            'Qloss': float(q_loss.detach()),
            'Ploss': float(actor_loss.detach()),
            'entropy': float(masked_mean(-log_prob.detach(), batch.mask).detach()),
        }
        if self.auto_alpha:
            alpha_loss = self.alpha_loss(log_prob, batch.mask)
            self._step(self.optimizer['alpha'], alpha_loss, [self.log_alpha])
            info['alpha'] = float(self.alpha())
        soft_update(self.target_q1, self.q1, self.tau)
        soft_update(self.target_q2, self.q2, self.tau)
        return info

    def critic_loss(self, batch: SequenceBatch) -> Tensor:
        target = self.target_value(batch)
        extended = extend_obs(batch.obs, batch.next_obs)
        predictions = []
        for critic in (self.q1, self.q2):
            features = features_from_history(
                critic.encoder, batch.prefix, batch.prefix_lengths, extended, grad=True
            )
            predictions.append(critic.head(features[:, :-1], batch.action).squeeze(-1))
        prediction = torch.stack(predictions, dim=0)
        per_q = F.smooth_l1_loss(prediction, target.detach().expand_as(prediction), reduction='none')
        return masked_mean(per_q.sum(0), batch.mask)

    @torch.no_grad()
    def target_value(self, batch: SequenceBatch) -> Tensor:
        extended = extend_obs(batch.obs, batch.next_obs)
        actor_features = features_from_history(
            self.actor.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
        )
        loc, scale = self.actor.head(actor_features[:, 1:])
        next_action, next_log_prob = sample_action(loc, scale, rsample=False)
        next_q = []
        for critic in (self.target_q1, self.target_q2):
            features = features_from_history(
                critic.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
            )
            next_q.append(critic.head(features[:, 1:], next_action).squeeze(-1))
        reward = batch.reward.squeeze(-1)
        # 发散才切断 bootstrap。线路终点是 truncated，环境会重置，但价值仍从重置前的下一观测自举。
        bootstrap = (~batch.terminated.reshape(reward.shape)).to(dtype=reward.dtype)
        return reward + self.gamma * bootstrap * (torch.min(next_q[0], next_q[1]) - self.alpha() * next_log_prob)

    def actor_loss(self, batch: SequenceBatch) -> tuple[Tensor, Tensor]:
        extended = extend_obs(batch.obs, batch.next_obs)
        features = features_from_history(self.actor.encoder, batch.prefix, batch.prefix_lengths, extended, grad=True)
        loc, scale = self.actor.head(features[:, :-1])
        action, log_prob = sample_action(loc, scale, rsample=True)
        # 冻结 Q 参数，但 Q 头仍对动作反传，actor 才能看到动作如何影响 Q。
        with frozen((self.q1, self.q2)):
            values = []
            for critic in (self.q1, self.q2):
                critic_features = features_from_history(
                    critic.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
                )
                values.append(critic.head(critic_features[:, :-1], action).squeeze(-1))
        loss = masked_mean(self.alpha() * log_prob - torch.min(values[0], values[1]), batch.mask)
        return loss, log_prob

    def alpha_loss(self, log_prob: Tensor, mask: Tensor) -> Tensor:
        return masked_mean(-self.log_alpha * (log_prob.detach() + self.target_entropy), mask)

    def _normalize(self, batch: SequenceBatch) -> SequenceBatch:
        if self.obs_norm is None or not self.obs_norm.enabled:
            return batch
        batch.prefix = self.obs_norm.normalize(batch.prefix)
        batch.obs = self.obs_norm.normalize(batch.obs)
        batch.next_obs = self.obs_norm.normalize(batch.next_obs)
        return batch

    def _step(self, optimizer, loss: Tensor, params) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(list(params), self.grad_clip_norm)
        optimizer.step()

    def state_dict(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'optimizers': {name: optimizer.state_dict() for name, optimizer in self.optimizer.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state['actor'])
        self.q1.load_state_dict(state['q1'])
        self.q2.load_state_dict(state['q2'])
        self.target_q1.load_state_dict(state['target_q1'])
        self.target_q2.load_state_dict(state['target_q2'])
        self.log_alpha.data.copy_(state['log_alpha'].to(self.log_alpha.device))
        for name, optimizer_state in state.get('optimizers', {}).items():
            if name in self.optimizer:
                self.optimizer[name].load_state_dict(optimizer_state)
