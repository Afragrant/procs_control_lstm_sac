"""Greedy Actor-Critic 的连续动作版本，网络和 SAC 共用。

Neumann et al., ICLR 2023。每个状态从 proposal 抽 N 个动作，按 Q 留下前 ρ 分位，
actor 只做这些动作的最大似然。熵加在 proposal 上，让它比 actor 收得慢。
Critic 是 Sarsa：下一动作来自 actor，目标里没有熵。双 Q 取较小值，避免高估排错精英集。
"""

from __future__ import annotations

import copy
import math

import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import functional as F

from algorithm.SAC.policy import OBS_NORMED_KEY, RunningObsNorm, build_actor, build_qvalue


def elite_actions(actions: Tensor, values: Tensor, rho: float) -> Tensor:
    """actions [N, B, A]、values [N, B]，返回每个状态 Q 最高的 ceil(ρN) 个动作 [B, h, A]。"""
    if actions.ndim != 3:
        raise ValueError(f'actions must be [N, B, A], got {tuple(actions.shape)}')
    n, batch, action_dim = actions.shape
    values = values.reshape(n, batch)
    if values.shape != (n, batch):
        raise ValueError(f'values must be [N, B], got {tuple(values.shape)} for actions {tuple(actions.shape)}')
    if not 0.0 < rho <= 1.0:
        raise ValueError(f'rho must be in (0, 1], got {rho}')
    keep = min(n, max(1, math.ceil(rho * n)))
    index = values.topk(keep, dim=0).indices  # [h, B]
    chosen = actions.gather(0, index.unsqueeze(-1).expand(-1, -1, action_dim))
    return chosen.transpose(0, 1).contiguous()


def _column(values: Tensor, batch: int) -> Tensor:
    return values.reshape(batch, -1)[:, :1]


class GreedySACLearner:
    def __init__(
        self,
        config,
        obs_shape: tuple[int, ...],
        action_dim: int,
        device: torch.device,
        obs_norm: RunningObsNorm | None = None,
    ):
        if bool(getattr(config, 'use_automatic_entropy_tuning', False)):
            raise ValueError('Greedy SAC keeps a fixed proposal entropy coefficient (alpha); turn automatic tuning off')
        self.config = config
        self.device = device
        self.obs_norm = obs_norm
        self.action_dim = int(action_dim)
        self.gamma = float(config.gamma)
        self.tau = float(config.tau)
        self.alpha = float(config.alpha)
        self.num_samples = int(getattr(config, 'cem_samples', 30))
        self.rho = float(getattr(config, 'cem_rho', 0.1))
        self.use_grad_clip = bool(getattr(config, 'use_grad_clip', False))
        self.grad_clip_norm = float(getattr(config, 'grad_clip_norm', 0.5))
        if self.num_samples < 1:
            raise ValueError(f'cem_samples must be positive, got {self.num_samples}')
        if not 0.0 < self.rho <= 1.0:
            raise ValueError(f'cem_rho must be in (0, 1], got {self.rho}')

        self.actor = build_actor(config.representation, obs_shape, action_dim, config, device)
        self.proposal = build_actor(config.representation, obs_shape, action_dim, config, device)
        self.proposal.load_state_dict(self.actor.state_dict())
        self.q1 = build_qvalue(config.representation, obs_shape, action_dim, config, device)
        self.q2 = build_qvalue(config.representation, obs_shape, action_dim, config, device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        actor_lr = float(config.learning_rate_actor)
        self.optimizer = {
            'actor': torch.optim.Adam(self.actor.parameters(), lr=actor_lr),
            'proposal': torch.optim.Adam(self.proposal.parameters(), lr=actor_lr),
            'critic': torch.optim.Adam(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                lr=float(config.learning_rate_critic),
            ),
        }

    def _normalize_batch(self, batch: TensorDict) -> TensorDict:
        obs = batch['observation']
        next_obs = batch['next', 'observation']
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize(obs)
            next_obs = self.obs_norm.normalize(next_obs)
        batch[OBS_NORMED_KEY] = obs
        batch['next', OBS_NORMED_KEY] = next_obs
        return batch

    def _q(self, net: nn.Module, obs: Tensor, action: Tensor) -> Tensor:
        batch = obs.shape[0]
        td = TensorDict({OBS_NORMED_KEY: obs, 'action': action}, batch_size=[batch], device=obs.device)
        return net(td)['state_action_value'].reshape(batch, 1)

    def _min_q(self, obs: Tensor, action: Tensor, *, target: bool) -> Tensor:
        q1 = self.q1_target if target else self.q1
        q2 = self.q2_target if target else self.q2
        return torch.min(self._q(q1, obs, action), self._q(q2, obs, action))

    def _candidate_values(self, obs: Tensor, actions: Tensor) -> Tensor:
        """actions [N, B, A] -> Q [N, B]，不回传到 Q。"""
        n, batch, _ = actions.shape
        flat_obs = obs.repeat_interleave(n, dim=0)
        flat_action = actions.transpose(0, 1).reshape(n * batch, self.action_dim)
        with torch.no_grad():
            values = self._min_q(flat_obs, flat_action, target=False)
        return values.reshape(batch, n).transpose(0, 1)

    def update(self, batch) -> dict[str, float]:
        batch = self._normalize_batch(batch.to(self.device))
        obs = batch[OBS_NORMED_KEY]
        next_obs = batch['next', OBS_NORMED_KEY]
        action = batch['action']
        batch_size = obs.shape[0]
        reward = _column(batch['next', 'reward'], batch_size)
        terminated = _column(batch['next', 'terminated'].to(dtype=reward.dtype), batch_size)

        with torch.no_grad():
            next_action = self.actor.get_dist(
                TensorDict({OBS_NORMED_KEY: next_obs}, batch_size=[batch_size], device=self.device)
            ).sample()
            target = reward + self.gamma * (1.0 - terminated) * self._min_q(next_obs, next_action, target=True)
        q_loss = F.mse_loss(self._q(self.q1, obs, action), target) + F.mse_loss(self._q(self.q2, obs, action), target)
        self._step('critic', q_loss, list(self.q1.parameters()) + list(self.q2.parameters()))
        _soft_update(self.q1_target, self.q1, self.tau)
        _soft_update(self.q2_target, self.q2, self.tau)

        obs_td = TensorDict({OBS_NORMED_KEY: obs}, batch_size=[batch_size], device=self.device)
        proposal_dist = self.proposal.get_dist(obs_td)
        candidates = proposal_dist.sample((self.num_samples,))
        if candidates.shape != (self.num_samples, batch_size, self.action_dim):
            raise RuntimeError(
                f'proposal sample shape {tuple(candidates.shape)} != '
                f'{(self.num_samples, batch_size, self.action_dim)}'
            )
        elites = elite_actions(candidates, self._candidate_values(obs, candidates), self.rho).detach()
        elites_nb = elites.transpose(0, 1)  # [h, B, A]，对齐分布的 sample 维

        actor_dist = self.actor.get_dist(obs_td)
        actor_loss = -actor_dist.log_prob(elites_nb).mean()
        self._step('actor', actor_loss, self.actor.parameters())

        proposal_logp = proposal_dist.log_prob(elites_nb)
        entropy = -proposal_dist.log_prob(proposal_dist.rsample())
        proposal_loss = -(proposal_logp.mean() + self.alpha * entropy.mean())
        self._step('proposal', proposal_loss, self.proposal.parameters())

        return {
            'Qloss': float(q_loss.detach()),
            'Ploss': float(actor_loss.detach()),
            'proposal_loss': float(proposal_loss.detach()),
            'alpha': self.alpha,
        }

    def _step(self, name: str, loss: Tensor, params) -> None:
        self.optimizer[name].zero_grad()
        loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        self.optimizer[name].step()

    def state_dict(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'proposal': self.proposal.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizer.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state['actor'])
        self.proposal.load_state_dict(state['proposal'])
        self.q1.load_state_dict(state['q1'])
        self.q2.load_state_dict(state['q2'])
        self.q1_target.load_state_dict(state['q1_target'])
        self.q2_target.load_state_dict(state['q2_target'])
        for name, opt_state in state.get('optimizers', {}).items():
            if name in self.optimizer:
                self.optimizer[name].load_state_dict(opt_state)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
