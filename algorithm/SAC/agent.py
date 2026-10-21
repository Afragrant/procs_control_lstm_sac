"""SAC 智能体，基于 TorchRL：Collector + ReplayBuffer + SACLoss 学习器。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .learner import SACLearner
from .policy import OBS_NORMED_KEY, RunningObsNorm, build_obs_norm_layer


class SACAgent:
    """离线策略 SAC 智能体。"""

    def __init__(self, config, envs: EnvBase | None = None):
        self.config = config
        self.device = torch.device(str(config.device))
        self.test_mode = bool(getattr(config, 'test_mode', False))

        # Pytorch 全局随机种子
        torch.manual_seed(int(getattr(config, 'seed', 0) or 0))

        if envs is None:
            raise ValueError('SACAgent requires envs to read observation/action specs')
        self.envs = envs
        # 并行环境数量
        self.num_envs = int(envs.num_workers)
        self.obs_shape = tuple(envs.observation_spec_unbatched['observation'].shape)
        self.action_dim = int(envs.action_spec_unbatched.shape[-1])

        # 观测归一化
        self.obs_norm = RunningObsNorm(
            self.obs_shape,
            getattr(config, 'obsnorm_range', 5.0),
            enabled=bool(getattr(config, 'use_obsnorm', True)),
        ).to(self.device)

        self.learner = self._make_learner()
        self.policy = self.learner.actor
        self.collector_policy = TensorDictSequential(
            build_obs_norm_layer(self.obs_norm),
            self.policy,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(int(config.buffer_size), device=torch.device('cpu')),
            batch_size=int(config.batch_size),
        )
        self._collector: Collector | None = None
        self._collector_iter = None
        self.current_step = 0
        self._vec_steps = 0
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_return_log: list[tuple[int, float]] = []

        log_dir = Path(str(config.log_dir)) / f'seed_{getattr(config, "seed", 0)}'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.model_dir_save = str(Path(str(config.model_dir)) / f'seed_{getattr(config, "seed", 0)}')

    def _make_learner(self):
        return SACLearner(self.config, self.obs_shape, self.action_dim, self.device, obs_norm=self.obs_norm)

    # 采集与训练
    def _build_collector(self) -> None:
        if self._collector is not None:
            return
        self._collector = Collector(
            self.envs,
            policy=self.collector_policy,
            frames_per_batch=self.num_envs,
            total_frames=-1,
            device=self.device,
            storing_device=torch.device('cpu'),
            auto_register_policy_transforms=False,
        )
        self._collector_iter = iter(self._collector)

    def train(self, vec_steps: int) -> dict[str, float]:
        """采集 vec_steps 个向量环境步，期间按调度做梯度更新，返回最后一次更新的指标。"""
        if self.test_mode:
            raise RuntimeError('train() is not allowed in test mode')
        self._build_collector()

        training_frequency = int(getattr(self.config, 'training_frequency', 1))
        n_epochs = int(getattr(self.config, 'n_epochs', 1))
        start_training = int(getattr(self.config, 'start_training', 0))
        info: dict[str, float] = {}
        collected_vec_steps = 0

        while collected_vec_steps < vec_steps:
            data = next(self._collector_iter)
            collected_vec_steps += 1
            self.current_step += data.numel()
            self._vec_steps += 1
            self._record_episode_returns(data)

            # buffer 存原始观测
            # 重新归一化 s 与 s'
            self.obs_norm.update(data['observation'])
            self.replay_buffer.extend(data.exclude(OBS_NORMED_KEY))

            if self.current_step > start_training and self._vec_steps % training_frequency == 0:
                for _ in range(n_epochs):
                    sample = self.replay_buffer.sample()
                    info = self.learner.update(sample)
                    for key, value in info.items():
                        self.writer.add_scalar(f'train/{key}', value, self.current_step)
        return info

    def _record_episode_returns(self, data) -> None:
        nxt = data['next']
        reward = np.asarray(nxt['reward'].detach().cpu().reshape(self.num_envs, -1)[:, 0], dtype=np.float64)
        terminated = nxt['terminated'].reshape(self.num_envs, -1)[:, 0]
        truncated = nxt['truncated'].reshape(self.num_envs, -1)[:, 0]
        done = np.asarray((terminated | truncated).detach().cpu(), dtype=bool)
        self._episode_return += reward
        if not done.any():
            return
        completed = self._episode_return[done]
        self.writer.add_scalar('train/episode_return', float(completed.mean()), self.current_step)
        for value in completed:
            self.episode_return_log.append((int(self.current_step), float(value)))
        self._episode_return[done] = 0.0

    # 推理接口（评估用）
    @torch.no_grad()
    def get_actions(self, observations: np.ndarray, test_mode: bool = False) -> dict[str, np.ndarray]:
        """输入原始观测，返回动作。test_mode=True 时为确定性动作（DETERMINISTIC 探索模式）。

        用当前统计量归一化到 OBS_NORMED_KEY 再过 actor，与采集层路径一致。
        """
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=torch.device('cpu'))
        single = obs.ndim == len(self.obs_shape)
        if single:
            obs = obs.unsqueeze(0)
        obs_normed = self.obs_norm.normalize(obs.to(self.device))
        td = TensorDict({OBS_NORMED_KEY: obs_normed}, batch_size=[obs_normed.shape[0]], device=obs_normed.device)
        if test_mode:
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td = self.policy(td)
        else:
            td = self.policy(td)
        return {'actions': td['action'].detach().cpu().numpy()}

    def finish(self) -> None:
        if self._collector is not None:
            try:
                self._collector.shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f'collector shutdown warning: {exc}')
            self._collector = None
            self._collector_iter = None
        self.writer.flush()
        self.writer.close()

    # 检查点
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
        self.policy.load_state_dict(checkpoint['policy'])
        self.obs_norm.load_state_dict(checkpoint['obs_norm'])
        self.learner.load_state_dict(checkpoint['learner'])
