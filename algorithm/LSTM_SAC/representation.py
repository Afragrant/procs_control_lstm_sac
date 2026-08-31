"""基于历史窗口的LSTM编码器"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

LSTM_HIDDEN = 256


class HistoryLSTM(nn.Module):
    """用单层 LSTM 把形状为 [batch, seq_len, obs_dim] 的观测窗口编码（成一个向量）。"""

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int] | None = None,
        device: str | int | torch.device | None = None,
        **kwargs,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f'HistoryLSTM expects input_shape (seq_len, obs_dim), got {tuple(input_shape)}')

        # seq_len：历史帧数, obs_dim：单帧观测维度
        seq_len, input_size = int(input_shape[0]), int(input_shape[-1])
        if seq_len <= 0:
            raise ValueError(f'HistoryLSTM expects a positive sequence length, got {seq_len}')
        if input_size <= 0:
            raise ValueError(f'HistoryLSTM expects a positive obs_dim, got {input_size}')

        hidden_size = int(hidden_sizes[-1]) if hidden_sizes else LSTM_HIDDEN
        self.input_shape = (seq_len, input_size)
        self.output_dim = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            device=device,
        )

    def forward(self, observations: Tensor) -> Tensor:
        # 不从 self.parameters() 取设备：SACLoss 的集成执行时参数会被临时换出；
        # 输入张量由 collector/learner 保证在正确设备上
        x = torch.as_tensor(observations, dtype=torch.float32)

        # 如果只传了单个样本 [seq_len, obs_dim]，自动补一个 batch 维变成 [1, seq_len, obs_dim]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim < 3 or x.shape[-2:] != self.input_shape:
            raise ValueError(f'HistoryLSTM expects trailing shape {self.input_shape}, got {tuple(x.shape)}')

        # (seq, dim) -> (batch, seq, dim)
        lead_shape = x.shape[:-2]
        x = x.reshape(-1, *self.input_shape)
        # (batch, seq, dim) -> (batch*seq, hidden_dim)
        output, _ = self.lstm(x)
        # (batch*seq, hidden_dim) -> (batch, seq, hidden_dim)
        return output[:, -1, :].reshape(*lead_shape, self.output_dim)  # 取最后一个时间步的隐藏状态作为输出
