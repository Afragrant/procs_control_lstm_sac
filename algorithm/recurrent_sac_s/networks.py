"""共享一条 LSTM。只有演员的前向会推进 (h, c)。"""

from __future__ import annotations

from torch import Tensor, nn
from torchrl.modules import TanhNormal

from algorithm.SAC.policy import make_action_head

from .encoder import RecurrentEncoder, State

__all__ = ['SharedRecurrentActor', 'action_distribution']


def action_distribution(loc: Tensor, scale: Tensor) -> TanhNormal:
    return TanhNormal(loc, scale, low=-1.0, high=1.0, tanh_loc=False)


class SharedRecurrentActor(nn.Module):
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
