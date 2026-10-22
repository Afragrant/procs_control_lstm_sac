"""按环境、按回合保存轨迹，并采样同一回合内的连续片段。"""

from __future__ import annotations

import torch
from torch import Tensor


class SequenceBatch:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def to(self, device) -> SequenceBatch:
        for name, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self


class TrajectoryBuffer:
    """容量按转移条数计。淘汰时只丢掉整条已结束回合，避免把前缀切碎。

    shortcut: 一条仍在进行的回合超过容量时会暂时留着，因为从回合起点重放前缀不能把它拆开。
    """

    def __init__(self, capacity: int, seq_len: int):
        if capacity <= 0 or seq_len <= 0:
            raise ValueError(f'capacity and seq_len must be positive, got {capacity}, {seq_len}')
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self._episodes: list[dict] = []
        self._open: dict[int, dict] = {}
        self._steps = 0
        self._next_episode_id = 0

    def start(self, num_envs: int) -> None:
        self._open = {index: self._blank() for index in range(int(num_envs))}

    def add(
        self,
        env_index: int,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        terminated: bool,
        truncated: bool,
    ) -> None:
        episode = self._open[env_index]
        episode['obs'].append(obs.detach().to(dtype=torch.float32, device='cpu').reshape(-1))
        episode['action'].append(action.detach().to(dtype=torch.float32, device='cpu').reshape(-1))
        episode['reward'].append(reward.detach().to(dtype=torch.float32, device='cpu').reshape(1))
        episode['next_obs'].append(next_obs.detach().to(dtype=torch.float32, device='cpu').reshape(-1))
        episode['terminated'].append(torch.tensor([bool(terminated)]))
        episode['truncated'].append(torch.tensor([bool(truncated)]))
        self._steps += 1
        if terminated or truncated:
            self._close(env_index)
        self._trim()

    @property
    def completed(self) -> list[dict]:
        return self._episodes

    def unfinished_frames(self, env_index: int) -> list[Tensor]:
        return self._open[env_index]['obs']

    def sample(self, batch_size: int) -> SequenceBatch | None:
        pool = self._candidates()
        if not pool:
            return None
        choice = torch.randint(len(pool), (int(batch_size),))
        prefixes = []
        lengths = []
        obs_rows = []
        action_rows = []
        reward_rows = []
        next_rows = []
        terminated_rows = []
        episode_ids = []
        step_ids = []
        masks = []
        for index in choice.tolist():
            episode = pool[index]
            start, segment, mask = self._window(episode)
            prefixes.append(episode['obs'][:start])
            lengths.append(start)
            obs_rows.append(segment['obs'])
            action_rows.append(segment['action'])
            reward_rows.append(segment['reward'])
            next_rows.append(segment['next_obs'])
            terminated_rows.append(segment['terminated'])
            masks.append(mask)
            episode_ids.append(int(episode['episode_id']))
            step_ids.append(torch.arange(start, start + self.seq_len))
        return SequenceBatch(
            prefix=_pad_prefix(prefixes),
            prefix_lengths=torch.tensor(lengths, dtype=torch.long),
            obs=torch.stack(obs_rows),
            action=torch.stack(action_rows),
            reward=torch.stack(reward_rows),
            next_obs=torch.stack(next_rows),
            terminated=torch.stack(terminated_rows),
            mask=torch.stack(masks),
            episode_id=torch.tensor(episode_ids, dtype=torch.long),
            step_id=torch.stack(step_ids),
        )

    def _window(self, episode: dict) -> tuple[int, dict, Tensor]:
        length = int(episode['obs'].shape[0])
        keys = ('obs', 'action', 'reward', 'next_obs', 'terminated')
        if length >= self.seq_len:
            start = int(torch.randint(length - self.seq_len + 1, ()).item())
            end = start + self.seq_len
            self._check_slice(episode, start, end)
            segment = {key: episode[key][start:end] for key in keys}
            return start, segment, torch.ones(self.seq_len)
        self._check_slice(episode, 0, length)
        segment = {key: _pad_time(episode[key], self.seq_len) for key in keys}
        # 补齐的第一帧放真实下一观测，终止步的 bootstrap 才能看到它。
        segment['obs'][length] = episode['next_obs'][-1]
        mask = torch.zeros(self.seq_len)
        mask[:length] = 1
        return 0, segment, mask

    def _candidates(self) -> list[dict]:
        ready = [episode for episode in self._episodes if int(episode['obs'].shape[0]) > 0]
        for episode in self._open.values():
            if len(episode['obs']) >= self.seq_len:
                ready.append(_stack(episode))
        return ready

    def _check_slice(self, episode: dict, start: int, end: int) -> None:
        obs = episode['obs'][start:end]
        nxt = episode['next_obs'][start:end]
        if end - start > 1 and not torch.equal(nxt[:-1], obs[1:]):
            raise RuntimeError('learning segment is not contiguous inside the episode')
        if end < episode['obs'].shape[0] and not torch.equal(nxt[-1], episode['obs'][end]):
            raise RuntimeError('next observation does not stay inside the same episode')
        if episode['terminated'][start : end - 1].any() or episode['truncated'][start : end - 1].any():
            raise RuntimeError('learning segment crosses an episode boundary')

    def _blank(self) -> dict:
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        return {
            'obs': [],
            'action': [],
            'reward': [],
            'next_obs': [],
            'terminated': [],
            'truncated': [],
            'episode_id': episode_id,
        }

    def _close(self, env_index: int) -> None:
        episode = self._open[env_index]
        if episode['obs']:
            self._episodes.append(_stack(episode))
        self._open[env_index] = self._blank()

    def _trim(self) -> None:
        while self._steps > self.capacity and self._episodes:
            dropped = self._episodes.pop(0)
            self._steps -= int(dropped['obs'].shape[0])


def _stack(episode: dict) -> dict:
    return {
        'obs': torch.stack(episode['obs']),
        'action': torch.stack(episode['action']),
        'reward': torch.stack(episode['reward']),
        'next_obs': torch.stack(episode['next_obs']),
        'terminated': torch.stack(episode['terminated']),
        'truncated': torch.stack(episode['truncated']),
        'episode_id': episode['episode_id'],
    }


def _pad_time(value: Tensor, seq_len: int) -> Tensor:
    if int(value.shape[0]) == seq_len:
        return value
    padded = value.new_zeros((seq_len, *value.shape[1:]))
    padded[: int(value.shape[0])] = value
    return padded


def _pad_prefix(prefixes: list[Tensor]) -> Tensor:
    batch = len(prefixes)
    width = max((int(prefix.shape[0]) for prefix in prefixes), default=0)
    obs_dim = int(prefixes[0].shape[-1])
    padded = prefixes[0].new_zeros(batch, width, obs_dim)
    for index, prefix in enumerate(prefixes):
        if prefix.shape[0]:
            padded[index, : prefix.shape[0]] = prefix
    return padded
