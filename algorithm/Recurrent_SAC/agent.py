"""采集时为每个并行环境保存 actor 的 (h, c)，回合结束只清该环境的槽位。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algorithm.SAC.policy import RunningObsNorm

from .encoder import replay_prefix, reset_env_slots, zero_state
from .learner import RecurrentSACLearner
from .networks import action_distribution
from .replay import TrajectoryBuffer, _pad_prefix


class RecurrentSACAgent:
    def __init__(self, config, envs):
        self.config = config
        self.device = torch.device(str(config.device))
        self.test_mode = bool(getattr(config, 'test_mode', False))
        torch.manual_seed(int(getattr(config, 'seed', 0) or 0))
        if envs is None:
            raise ValueError('RecurrentSACAgent requires envs')
        self.envs = envs
        self.num_envs = int(envs.num_workers)
        self.obs_shape = tuple(envs.observation_spec_unbatched['observation'].shape)
        if len(self.obs_shape) != 1:
            raise ValueError(f'Recurrent SAC expects a single-frame observation, got {self.obs_shape}')
        self.obs_dim = int(self.obs_shape[0])
        self.action_dim = int(envs.action_spec_unbatched.shape[-1])

        self.obs_norm = RunningObsNorm(
            self.obs_shape,
            getattr(config, 'obsnorm_range', 5.0),
            enabled=bool(getattr(config, 'use_obsnorm', True)),
        ).to(self.device)
        self.learner = self._make_learner()
        self.policy = self.learner.actor
        self.hidden_size = self.policy.encoder.hidden_size
        self.num_layers = self.policy.encoder.num_layers
        self.replay = TrajectoryBuffer(int(config.buffer_size), int(config.sequence_length))
        self.current_step = 0
        self._vec_steps = 0
        self._td = None
        self._actor_state = None
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_return_log: list[tuple[int, float]] = []

        log_dir = Path(str(config.log_dir)) / f'seed_{getattr(config, "seed", 0)}'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.model_dir_save = str(Path(str(config.model_dir)) / f'seed_{getattr(config, "seed", 0)}')

    def _make_learner(self):
        return RecurrentSACLearner(self.config, self.obs_dim, self.action_dim, self.device, self.obs_norm)

    def train(self, vec_steps: int) -> dict[str, float]:
        if self.test_mode:
            raise RuntimeError('train() is not allowed in test mode')
        self._ensure_started()
        training_frequency = int(getattr(self.config, 'training_frequency', 1))
        n_epochs = int(getattr(self.config, 'n_epochs', 1))
        start_training = int(getattr(self.config, 'start_training', 0))
        info: dict[str, float] = {}
        for _ in range(int(vec_steps)):
            self._interact_once()
            if self.current_step > start_training and self._vec_steps % training_frequency == 0:
                for _ in range(n_epochs):
                    sample = self.replay.sample(int(self.config.batch_size))
                    if sample is None:
                        continue
                    info = self.learner.update(sample)
                    self._refresh_actor_state()
                    self.writer.add_scalar('train/policy_loss', info['Ploss'], self.current_step)
                    self.writer.add_scalar('train/q_loss', info['Qloss'], self.current_step)
                    self.writer.add_scalar('train/entropy', info['entropy'], self.current_step)
                    if 'proposal_loss' in info:
                        self.writer.add_scalar('train/proposal_loss', info['proposal_loss'], self.current_step)
                    if 'alpha' in info:
                        self.writer.add_scalar('train/alpha', info['alpha'], self.current_step)
        return info

    def _zero_state(self, batch: int):
        return zero_state(batch, self.hidden_size, self.device, num_layers=self.num_layers)

    def _ensure_started(self) -> None:
        if self._td is not None:
            return
        self._td = self.envs.reset()
        self._actor_state = self._zero_state(self.num_envs)
        self.replay.start(self.num_envs)

    @torch.no_grad()
    def _refresh_actor_state(self) -> None:
        # 缓冲区里的动作来自更新前的权重。从回合起点重放，下一次采集才和训练用的 (h, c) 一致。
        rows = []
        lengths = []
        for index in range(self.num_envs):
            frames = self.replay.unfinished_frames(index)
            rows.append(torch.stack(frames) if frames else torch.zeros(0, self.obs_dim))
            lengths.append(int(rows[-1].shape[0]))
        prefix = self.obs_norm.normalize(_pad_prefix(rows).to(self.device))
        self._actor_state = replay_prefix(self.policy.encoder, prefix, torch.tensor(lengths))

    def _interact_once(self) -> None:
        obs = self._td['observation']
        action, self._actor_state = self.act(obs, self._actor_state, deterministic=False)
        self._td.set('action', action.detach().to(device=self._td.device, dtype=torch.float32))
        transition, self._td = self.envs.step_and_maybe_reset(self._td)
        warmup = int(getattr(self.config, 'start_training', 0))
        warming = warmup <= 0 or self.current_step < warmup
        if warming:
            self.obs_norm.update(transition['observation'])
        self._store_transition(transition)
        self._record_episode_returns(transition)
        # 统计量变过之后，在线 (h, c) 仍按旧归一化累加。存完这一帧再重放，下一步才和回放一致。
        if warming and self.obs_norm.enabled:
            self._refresh_actor_state()
        # 环境 reset 之后、下一帧进入 actor 之前清零。采集 batch 结束不清。
        done = self._done_mask(transition)
        self._actor_state = reset_env_slots(self._actor_state, done)
        self.current_step += self.num_envs
        self._vec_steps += 1

    def _store_transition(self, transition) -> None:
        nxt = transition['next']
        for index in range(self.num_envs):
            terminated = bool(nxt['terminated'][index].reshape(-1)[0].item())
            truncated = bool(nxt['truncated'][index].reshape(-1)[0].item())
            self.replay.add(
                index,
                transition['observation'][index],
                transition['action'][index],
                nxt['reward'][index],
                nxt['observation'][index],
                terminated,
                truncated,
            )

    def _done_mask(self, transition) -> torch.Tensor:
        nxt = transition['next']
        terminated = nxt['terminated'].reshape(self.num_envs, -1)[:, 0]
        truncated = nxt['truncated'].reshape(self.num_envs, -1)[:, 0]
        return terminated | truncated

    def _record_episode_returns(self, transition) -> None:
        nxt = transition['next']
        reward = np.asarray(nxt['reward'].detach().cpu().reshape(self.num_envs, -1)[:, 0], dtype=np.float64)
        done = np.asarray(self._done_mask(transition).detach().cpu(), dtype=bool)
        self._episode_return += reward
        if not done.any():
            return
        step = int(self.current_step + self.num_envs)
        completed = self._episode_return[done]
        self.writer.add_scalar('train/episode_return', float(completed.mean()), step)
        for value in completed:
            self.episode_return_log.append((step, float(value)))
        self._episode_return[done] = 0.0

    @torch.no_grad()
    def act(self, obs, state, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim != 2:
            raise ValueError(f'act expects [batch, obs_dim], got {tuple(obs_t.shape)}')
        if state[0].shape[1] != obs_t.shape[0]:
            raise ValueError(f'state batch {state[0].shape[1]} does not match observations {obs_t.shape[0]}')
        normed = self.obs_norm.normalize(obs_t)
        loc, scale, next_state = self.policy(normed.unsqueeze(1), state)
        dist = action_distribution(loc[:, 0], scale[:, 0])
        action = dist.deterministic_sample if deterministic else dist.sample()
        return action, next_state

    def get_actions(self, observations: np.ndarray, test_mode: bool = False, state=None) -> dict:
        """不传入 state 时从零状态算动作，不会读写采集用的 actor 状态。"""
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32))
        if obs.ndim == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        if state is None:
            state = self._zero_state(obs.shape[0])
        action, next_state = self.act(obs, state, deterministic=test_mode)
        return {'actions': action.detach().cpu().numpy(), 'state': next_state}

    def eval_action(self):
        """评估用的独立状态。和采集状态不是同一份。"""
        holder = {'state': self._zero_state(1)}

        def reset_state():
            holder['state'] = self._zero_state(1)

        def _fn(obs):
            result = self.get_actions(obs, test_mode=True, state=holder['state'])
            holder['state'] = result['state']
            return result['actions'][0]

        _fn.reset_state = reset_state
        return _fn

    def finish(self) -> None:
        self.writer.flush()
        self.writer.close()

    def save_model(self, name: str, model_path: str | None = None) -> None:
        save_dir = model_path or self.model_dir_save
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'policy': self.policy.state_dict(),
            'obs_norm': self.obs_norm.state_dict(),
            'learner': self.learner.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, name))

    def load_model(self, path: str, model: str | None = None) -> None:
        checkpoint_path = Path(path) if model is None else Path(path) / model
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.learner.load_state_dict(checkpoint['learner'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.obs_norm.load_state_dict(checkpoint['obs_norm'])
