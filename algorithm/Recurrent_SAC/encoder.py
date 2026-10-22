"""显式传入和返回 (h, c) 的双层 LSTM 编码器。模块内部不保存隐状态。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

State = tuple[Tensor, Tensor]
FORGET_BIAS = 1.0
NUM_LAYERS = 2


def set_forget_bias(lstm: nn.LSTM, value: float = FORGET_BIAS) -> None:
    """遗忘门总偏置置为 value。bias_ih 与 bias_hh 在门里相加，只把前者设为 value，后者清零。"""
    hidden = int(lstm.hidden_size)
    with torch.no_grad():
        for name, param in lstm.named_parameters():
            if param.ndim != 1 or param.shape[0] != 4 * hidden:
                continue
            forget = param[hidden : 2 * hidden]
            if 'bias_ih' in name:
                forget.fill_(value)
            elif 'bias_hh' in name:
                forget.zero_()


def _lstm(lstm: nn.LSTM, *args):
    # 优化器原地改写权重后，cuDNN 的连续权重块会失效。先压紧，否则每次前向都告警并多拷一次。
    lstm.flatten_parameters()
    return lstm(*args)


class RecurrentEncoder(nn.Module):
    """把观测序列编码成每个时刻的特征，并拼上当前观测。

    obs_seq 是 [batch, time, obs_dim]。state 是双层 LSTM 的 (h, c)，形状 [2, batch, hidden]。
    输出只用最后一层，所以特征维仍是 hidden + obs_dim。
    """

    def __init__(self, obs_dim: int, hidden_size: int, device=None):
        super().__init__()
        obs_dim = int(obs_dim)
        hidden_size = int(hidden_size)
        if obs_dim <= 0 or hidden_size <= 0:
            raise ValueError(f'obs_dim and hidden_size must be positive, got {obs_dim}, {hidden_size}')
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.num_layers = NUM_LAYERS
        self.output_dim = hidden_size + obs_dim
        self.lstm = nn.LSTM(
            obs_dim, hidden_size, num_layers=self.num_layers, batch_first=True, device=device
        )
        set_forget_bias(self.lstm)

    def forward(self, obs_seq: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        if obs_seq.ndim != 3 or obs_seq.shape[-1] != self.obs_dim:
            raise ValueError(f'expected [batch, time, {self.obs_dim}], got {tuple(obs_seq.shape)}')
        output, next_state = _lstm(self.lstm, obs_seq, state)
        return torch.cat([output, obs_seq], dim=-1), next_state


def zero_state(
    batch: int, hidden_size: int, device, dtype=torch.float32, num_layers: int = NUM_LAYERS
) -> State:
    hidden = torch.zeros(int(num_layers), int(batch), int(hidden_size), device=device, dtype=dtype)
    return hidden, torch.zeros_like(hidden)


def reset_env_slots(state: State, done: Tensor) -> State:
    """只清零 done 为真的环境槽位。done 形状 [num_envs]。"""
    done = done.reshape(-1).bool()
    hidden, cell = state
    if hidden.shape[1] != done.shape[0] or cell.shape != hidden.shape:
        raise ValueError(f'state batch {hidden.shape[1]} does not match done {tuple(done.shape)}')
    done = done.to(device=hidden.device)
    if not bool(done.any()):
        return state
    hidden = hidden.clone()
    cell = cell.clone()
    hidden[:, done] = 0
    cell[:, done] = 0
    return hidden, cell


def replay_prefix(encoder: RecurrentEncoder, prefix: Tensor, lengths: Tensor) -> State:
    """从零状态重放每个样本自己的前缀，得到学习段起点的 (h, c)。长度为 0 的样本保持零状态。"""
    batch, time, _ = prefix.shape
    lengths = lengths.detach().to(dtype=torch.long, device='cpu').reshape(-1)
    if lengths.numel() != batch:
        raise ValueError(f'prefix batch {batch} does not match lengths {lengths.numel()}')
    state = zero_state(batch, encoder.hidden_size, prefix.device, prefix.dtype, encoder.num_layers)
    if time == 0 or not bool((lengths > 0).any()):
        return state
    if int(lengths.max()) > time:
        raise ValueError(f'prefix length {int(lengths.max())} exceeds time dimension {time}')
    index = (lengths > 0).nonzero(as_tuple=False).flatten()
    packed = nn.utils.rnn.pack_padded_sequence(
        prefix.index_select(0, index.to(prefix.device)),
        lengths.index_select(0, index),
        batch_first=True,
        enforce_sorted=False,
    )
    _, (hidden, cell) = _lstm(encoder.lstm, packed)
    state_h, state_c = state
    state_h = state_h.clone()
    state_c = state_c.clone()
    slot = index.to(state_h.device)
    state_h[:, slot] = hidden
    state_c[:, slot] = cell
    return state_h, state_c
