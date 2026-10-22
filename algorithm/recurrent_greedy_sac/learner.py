"""序列上的条件 CEM。Critic 是不带熵的 Sarsa，下一动作来自 actor。

精英只在有效时间步上筛选。候选动作只进 Q 头，不重跑 LSTM，也不写回历史。
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from algorithm.Greedy_SAC.learner import elite_actions
from algorithm.Recurrent_SAC.learner import (
    _hidden_size,
    extend_obs,
    features_from_history,
    masked_mean,
    sample_action,
    soft_update,
)
from algorithm.Recurrent_SAC.networks import RecurrentActor, RecurrentQ, action_distribution
from algorithm.Recurrent_SAC.replay import SequenceBatch
from algorithm.SAC.policy import RunningObsNorm


def sequence_elites(actions: Tensor, values: Tensor, mask: Tensor, rho: float) -> Tensor:
    """每个有效时间步单独留精英。actions [N, B, L, A]，values [N, B, L]，返回 [K, B, L, A]。

    padding 位置保持 0，不参与排序。batch 和时间不会被排到同一次 topk 里。
    """
    if actions.ndim != 4:
        raise ValueError(f'actions must be [N, B, L, A], got {tuple(actions.shape)}')
    n, batch, length, _ = actions.shape
    if values.shape != (n, batch, length):
        raise ValueError(f'values must be [N, B, L], got {tuple(values.shape)} for actions {tuple(actions.shape)}')
    if mask.shape != (batch, length):
        raise ValueError(f'mask must be [B, L], got {tuple(mask.shape)}')
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError('no valid steps')
    flat_actions = actions.permute(1, 2, 0, 3)[valid].permute(1, 0, 2).contiguous()
    flat_values = values.permute(1, 2, 0)[valid].transpose(0, 1).contiguous()
    chosen = elite_actions(flat_actions, flat_values, rho).detach()
    return _scatter(chosen, valid)


def _scatter(chosen: Tensor, valid: Tensor) -> Tensor:
    """chosen [M, K, A]，valid [B, L] -> [K, B, L, A]。"""
    if int(valid.sum()) != chosen.shape[0]:
        raise ValueError(f'valid steps {int(valid.sum())} != elite rows {chosen.shape[0]}')
    batch, length = valid.shape
    keep, action_dim = chosen.shape[1], chosen.shape[2]
    placed = chosen.new_zeros(batch, length, keep, action_dim)
    placed[valid] = chosen
    return placed.permute(2, 0, 1, 3).contiguous()


class RecurrentGreedySACLearner:
    def __init__(self, config, obs_dim: int, action_dim: int, device: torch.device, obs_norm: RunningObsNorm | None):
        if bool(getattr(config, 'use_automatic_entropy_tuning', False)):
            raise ValueError(
                'Recurrent Greedy SAC keeps a fixed proposal entropy coefficient (alpha); turn automatic tuning off'
            )
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

        hidden = _hidden_size(config)
        actor_sizes = list(config.actor_hidden_size)
        critic_sizes = list(config.critic_hidden_size)
        activation = config.activation
        self.actor = RecurrentActor(obs_dim, action_dim, hidden, actor_sizes, activation, device)
        self.proposal = RecurrentActor(obs_dim, action_dim, hidden, actor_sizes, activation, device)
        self.proposal.load_state_dict(self.actor.state_dict())
        self.q1 = RecurrentQ(obs_dim, action_dim, hidden, critic_sizes, activation, device)
        self.q2 = RecurrentQ(obs_dim, action_dim, hidden, critic_sizes, activation, device)
        self.target_q1 = copy.deepcopy(self.q1).eval()
        self.target_q2 = copy.deepcopy(self.q2).eval()
        for target in (self.target_q1, self.target_q2):
            for param in target.parameters():
                param.requires_grad_(False)

        actor_lr = float(config.learning_rate_actor)
        self.optimizer = {
            'actor': torch.optim.Adam(self.actor.parameters(), lr=actor_lr),
            'proposal': torch.optim.Adam(self.proposal.parameters(), lr=actor_lr),
            'critic': torch.optim.Adam(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                lr=float(config.learning_rate_critic),
            ),
        }

    def update(self, batch: SequenceBatch) -> dict[str, float]:
        batch = self._normalize(batch.to(self.device))
        q_loss = self.critic_loss(batch)
        self._step('critic', q_loss, list(self.q1.parameters()) + list(self.q2.parameters()))
        elites = self.elites(batch)
        actor_loss = self.actor_loss(batch, elites)
        self._step('actor', actor_loss, self.actor.parameters())
        proposal_loss, entropy = self.proposal_loss(batch, elites)
        self._step('proposal', proposal_loss, self.proposal.parameters())
        soft_update(self.target_q1, self.q1, self.tau)
        soft_update(self.target_q2, self.q2, self.tau)
        return {
            'Qloss': float(q_loss.detach()),
            'Ploss': float(actor_loss.detach()),
            'proposal_loss': float(proposal_loss.detach()),
            'entropy': float(entropy.detach()),
            'alpha': self.alpha,
        }

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
        per_q = F.mse_loss(prediction, target.detach().expand_as(prediction), reduction='none')
        return masked_mean(per_q.sum(0), batch.mask)

    @torch.no_grad()
    def target_value(self, batch: SequenceBatch) -> Tensor:
        extended = extend_obs(batch.obs, batch.next_obs)
        actor_features = features_from_history(
            self.actor.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
        )
        loc, scale = self.actor.head(actor_features[:, 1:])
        next_action, _ = sample_action(loc, scale, rsample=False)
        next_q = []
        for critic in (self.target_q1, self.target_q2):
            features = features_from_history(
                critic.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
            )
            next_q.append(critic.head(features[:, 1:], next_action).squeeze(-1))
        reward = batch.reward.squeeze(-1)
        # 只有 terminated 切断 bootstrap。truncated 仍从重置前的下一观测自举。
        bootstrap = (~batch.terminated.reshape(reward.shape)).to(dtype=reward.dtype)
        return reward + self.gamma * bootstrap * torch.min(next_q[0], next_q[1])

    @torch.no_grad()
    def elites(self, batch: SequenceBatch) -> Tensor:
        valid = self._valid(batch)
        extended = extend_obs(batch.obs, batch.next_obs)
        features = features_from_history(
            self.proposal.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
        )
        loc, scale = self.proposal.head(features[:, :-1][valid])
        candidates = action_distribution(loc, scale).sample((self.num_samples,))
        count = int(valid.sum())
        if candidates.shape != (self.num_samples, count, self.action_dim):
            raise RuntimeError(
                f'proposal sample shape {tuple(candidates.shape)} != '
                f'{(self.num_samples, count, self.action_dim)}'
            )
        values = self._candidate_q(batch, extended, valid, candidates)
        chosen = elite_actions(candidates, values, self.rho).detach()
        return _scatter(chosen, valid)

    def _candidate_q(self, batch: SequenceBatch, extended: Tensor, valid: Tensor, candidates: Tensor) -> Tensor:
        """candidates [N, M, A] -> 在线双 Q 的较小值 [N, M]。历史特征只算一次。"""
        # shortcut: 所有有效步一次送进 Q 头；显存不够时再按时间分块。
        n, count, _ = candidates.shape
        scores = []
        for critic in (self.q1, self.q2):
            features = features_from_history(
                critic.encoder, batch.prefix, batch.prefix_lengths, extended, grad=False
            )
            feat = features[:, :-1][valid].unsqueeze(0).expand(n, -1, -1)
            score = critic.head(feat, candidates).squeeze(-1)
            if score.shape != (n, count):
                raise RuntimeError(f'candidate Q shape {tuple(score.shape)} != {(n, count)}')
            scores.append(score)
        return torch.min(scores[0], scores[1])

    def actor_loss(self, batch: SequenceBatch, elites: Tensor) -> Tensor:
        elites = self._checked_elites(batch, elites)
        log_prob = self._distribution(self.actor, batch).log_prob(elites)
        if log_prob.shape != elites.shape[:-1]:
            raise RuntimeError(f'actor log_prob shape {tuple(log_prob.shape)} != {tuple(elites.shape[:-1])}')
        return -masked_mean(log_prob.mean(dim=0), batch.mask)

    def proposal_loss(self, batch: SequenceBatch, elites: Tensor) -> tuple[Tensor, Tensor]:
        elites = self._checked_elites(batch, elites)
        dist = self._distribution(self.proposal, batch)
        log_prob = dist.log_prob(elites)
        entropy = -dist.log_prob(dist.rsample())
        if log_prob.shape != elites.shape[:-1] or entropy.shape != batch.mask.shape:
            raise RuntimeError(
                f'proposal log_prob {tuple(log_prob.shape)} entropy {tuple(entropy.shape)} '
                f'do not match elites {tuple(elites.shape)} mask {tuple(batch.mask.shape)}'
            )
        loss = -masked_mean(log_prob.mean(dim=0) + self.alpha * entropy, batch.mask)
        return loss, masked_mean(entropy, batch.mask).detach()

    def _distribution(self, network: RecurrentActor, batch: SequenceBatch):
        extended = extend_obs(batch.obs, batch.next_obs)
        features = features_from_history(
            network.encoder, batch.prefix, batch.prefix_lengths, extended, grad=True
        )
        loc, scale = network.head(features[:, :-1])
        return action_distribution(loc, scale)

    def _checked_elites(self, batch: SequenceBatch, elites: Tensor) -> Tensor:
        expected = (batch.mask.shape[0], batch.mask.shape[1], self.action_dim)
        if elites.ndim != 4 or elites.shape[1:] != expected:
            raise ValueError(f'elites must be [K, B, L, A] = [K, {expected}], got {tuple(elites.shape)}')
        return elites.detach()

    def _valid(self, batch: SequenceBatch) -> Tensor:
        if batch.mask.ndim != 2:
            raise ValueError(f'mask must be [B, L], got {tuple(batch.mask.shape)}')
        valid = batch.mask.bool()
        if not bool(valid.any()):
            raise ValueError('sequence batch has no valid steps')
        return valid

    def _normalize(self, batch: SequenceBatch) -> SequenceBatch:
        if self.obs_norm is None or not self.obs_norm.enabled:
            return batch
        batch.prefix = self.obs_norm.normalize(batch.prefix)
        batch.obs = self.obs_norm.normalize(batch.obs)
        batch.next_obs = self.obs_norm.normalize(batch.next_obs)
        return batch

    def _step(self, name: str, loss: Tensor, params) -> None:
        self.optimizer[name].zero_grad(set_to_none=True)
        loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(list(params), self.grad_clip_norm)
        self.optimizer[name].step()

    def state_dict(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'proposal': self.proposal.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'optimizers': {name: optimizer.state_dict() for name, optimizer in self.optimizer.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state['actor'])
        self.proposal.load_state_dict(state['proposal'])
        self.q1.load_state_dict(state['q1'])
        self.q2.load_state_dict(state['q2'])
        self.target_q1.load_state_dict(state['target_q1'])
        self.target_q2.load_state_dict(state['target_q2'])
        for name, optimizer_state in state.get('optimizers', {}).items():
            if name in self.optimizer:
                self.optimizer[name].load_state_dict(optimizer_state)
