"""Actor 与 Q 各自持有编码器。候选动作只进 Q 头，不写进历史。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchrl.modules import TanhNormal

from algorithm.SAC.policy import QHead, make_action_head

from .encoder import RecurrentEncoder, State


def action_distribution(loc: Tensor, scale: Tensor) -> TanhNormal:
    return TanhNormal(loc, scale, low=-1.0, high=1.0, tanh_loc=False)


class RecurrentActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        head_sizes: list[int],
        activation: str,
        device=None,
    ):
        super().__init__()
        self.encoder = RecurrentEncoder(obs_dim, hidden_size, device)
        self.head = make_action_head(self.encoder.output_dim, action_dim, head_sizes, activation, device)

    def forward(self, obs_seq: Tensor, state: State | None = None) -> tuple[Tensor, Tensor, State]:
        features, next_state = self.encoder(obs_seq, state)
        loc, scale = self.head(features)
        return loc, scale, next_state


class RecurrentQ(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        head_sizes: list[int],
        activation: str,
        device=None,
    ):
        super().__init__()
        self.encoder = RecurrentEncoder(obs_dim, hidden_size, device)
        self.head = QHead(self.encoder.output_dim, action_dim, head_sizes, activation, device)

    def forward(self, obs_seq: Tensor, action: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        features, next_state = self.encoder(obs_seq, state)
        return self.head(features, action), next_state
