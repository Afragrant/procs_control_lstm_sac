"""History-window MLP representation for MLP-SAC (POMDP 帧堆叠 + 展平 MLP 编码器)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from algorithm.SAC.policy import make_activation

DEFAULT_MLP_HIDDEN = 256


class HistoryMLP(nn.Module):
    """Encode a [batch, seq_len, obs_dim] observation window by flattening it through an MLP.

    与 Basic_MLP（1-D 观测）不同，堆叠窗口先展平为 seq_len * obs_dim 再过全连接层，
    帧顺序由输入位置携带。与 HistoryLSTM 共享同一堆叠观测接口。
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int] | None = None,
        activation: str = 'relu',
        device: str | int | torch.device | None = None,
        **kwargs,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f'HistoryMLP expects input_shape (seq_len, obs_dim), got {tuple(input_shape)}')

        seq_len, obs_dim = int(input_shape[0]), int(input_shape[1])
        if seq_len <= 0 or obs_dim <= 0:
            raise ValueError(f'HistoryMLP expects positive seq_len and obs_dim, got {tuple(input_shape)}')

        hidden_sizes = tuple(hidden_sizes) if hidden_sizes else (DEFAULT_MLP_HIDDEN,)
        self.input_shape = (seq_len, obs_dim)
        self.output_dim = int(hidden_sizes[-1])

        layers: list[nn.Module] = []
        in_dim = seq_len * obs_dim
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_dim, hidden, device=device), make_activation(activation)]
            in_dim = hidden
        self.model = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        seq_len, obs_dim = self.input_shape
        # 不从 self.parameters() 取设备：SACLoss 的 vmap 集成执行时参数会被临时换出；
        # 输入张量由 collector/learner 保证在正确设备上
        x = torch.as_tensor(observations, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim < 3 or x.shape[-2:] != self.input_shape:
            raise ValueError(f'HistoryMLP expects trailing shape {self.input_shape}, got {tuple(x.shape)}')
        # pseudo-vmap 集成会带来额外的前置维，压平进 batch 处理
        lead_shape = x.shape[:-2]
        x = x.reshape(-1, *self.input_shape)
        return self.model(x.flatten(start_dim=1)).reshape(*lead_shape, self.output_dim)
