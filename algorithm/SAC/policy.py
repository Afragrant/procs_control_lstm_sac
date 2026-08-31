"""SAC 网络构建"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import Tensor, nn
from torchrl.data import Bounded
from torchrl.modules import ProbabilisticActor, TanhNormal

# 数值稳定常数
OBS_NORM_EPS = 1e-8
OBS_NORMED_KEY = 'observation_normed'

# 激活函数
ACTIVATIONS: dict[str, type[nn.Module]] = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'silu': nn.SiLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
}


def make_activation(name: str) -> nn.Module:
    """
    创建激活函数
    """
    cls = ACTIVATIONS.get(str(name).lower())
    if cls is None:
        raise ValueError(f'unknown activation {name!r}, expected one of {sorted(ACTIVATIONS)}')
    return cls()


class RunningObsNorm(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], clip_range: float = 5.0, device=None, enabled: bool = True):
        super().__init__()
        if not obs_shape:
            raise ValueError('obs_shape must be non-empty')
        self.obs_shape = tuple(obs_shape)
        self.feat_shape = (int(obs_shape[-1]),)
        self._stacked_window = len(self.obs_shape) == 2
        self.clip_range = float(clip_range)
        self.enabled = bool(enabled)
        self.register_buffer('mean', torch.zeros(self.feat_shape, device=device))
        self.register_buffer('var', torch.ones(self.feat_shape, device=device))
        self.register_buffer('count', torch.tensor(1e-4, device=device))

    @torch.no_grad()
    def update(self, obs: Tensor) -> None:
        if not self.enabled:
            return
        obs = obs.detach().to(self.mean.device)
        if self._stacked_window:
            obs = obs[..., -1, :]
        flat = obs.reshape(-1, *self.feat_shape).to(torch.float64)
        n = flat.shape[0]
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = float(n)

        delta = batch_mean - self.mean.double()
        total = self.count.item() + batch_count
        new_mean = self.mean.double() + delta * batch_count / total
        m_a = self.var.double() * self.count.item()
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count.item() * batch_count / total
        self.mean.copy_(new_mean.float())
        self.var.copy_((m2 / total).float())
        self.count.fill_(total)

    def normalize(self, obs: Tensor) -> Tensor:
        if not self.enabled:
            return obs
        normalized = (obs - self.mean) / torch.sqrt(self.var + OBS_NORM_EPS)
        return normalized.clamp(-self.clip_range, self.clip_range)

    def forward(self, obs: Tensor) -> Tensor:
        return self.normalize(obs)


def build_obs_norm_layer(obs_norm: RunningObsNorm) -> TensorDictModule:
    """采集策略的归一化包装层：把原始 'observation' 归一化后写入 OBS_NORMED_KEY。

    不改写 'observation' 本身——buffer 需要保留原始观测，供更新时用同一套
    统计量重新归一化 s 与 s'。
    """
    return TensorDictModule(obs_norm, in_keys=['observation'], out_keys=[OBS_NORMED_KEY])


def make_action_head(
    input_dim: int, action_dim: int, hidden_sizes: list[int], activation: str, device=None
) -> nn.Module:
    """actor 头部：隐藏层 + (loc, scale) 参数提取。"""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden in hidden_sizes:
        layers += [nn.Linear(in_dim, hidden, device=device), make_activation(activation)]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 2 * action_dim, device=device), NormalParamExtractor()]
    return nn.Sequential(*layers)


class QHead(nn.Module):
    """critic 头部：拼接特征与动作后输出 Q 值。"""

    def __init__(self, feat_dim: int, action_dim: int, hidden_sizes: list[int], activation: str, device=None):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feat_dim + action_dim
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_dim, hidden, device=device), make_activation(activation)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1, device=device)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([feat, action], dim=-1))


def _build_representation(representation: str, obs_shape: tuple[int, ...], config, device) -> nn.Module:
    """按配置构建表示网络。环境为 POMDP：两种表示都以堆叠窗口 (seq_len, obs_dim) 为输入。"""
    if len(obs_shape) != 2:
        raise ValueError(f'representation {representation!r} expects (seq_len, obs_dim) observation, got {obs_shape}')
    if representation == 'History_MLP':
        from algorithm.MLP_SAC import HistoryMLP

        return HistoryMLP(obs_shape, list(config.representation_hidden_size), config.activation, device)
    if representation == 'History_LSTM':
        from algorithm.LSTM_SAC import HistoryLSTM

        return HistoryLSTM(obs_shape, list(config.representation_hidden_size), device=device)
    raise ValueError(f'unknown representation {representation!r} (expected History_MLP or History_LSTM)')


def build_actor(
    representation: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    config,
    device: torch.device,
    obs_key: str = OBS_NORMED_KEY,
) -> ProbabilisticActor:
    """构建 TorchRL 概率 actor：TanhNormal，动作范围 [-1, 1]。

    网络本身不做归一化，从 obs_key（默认 OBS_NORMED_KEY）读取已归一化观测——
    该键由采集策略包装层 / learner batch 预处理 / get_actions 各自填充。
    """
    repr_net = _build_representation(representation, obs_shape, config, device)
    feat_dim = repr_net.output_dim

    actor_module = TensorDictModule(
        nn.Sequential(
            repr_net,
            make_action_head(feat_dim, action_dim, list(config.actor_hidden_size), config.activation, device),
        ),
        in_keys=[obs_key],
        out_keys=['loc', 'scale'],
    )
    return ProbabilisticActor(
        spec=Bounded(-1.0, 1.0, (action_dim,), device=device),
        in_keys=['loc', 'scale'],
        module=actor_module,
        distribution_class=TanhNormal,
        distribution_kwargs={'low': -1.0, 'high': 1.0, 'tanh_loc': False},
        return_log_prob=False,
    )


def build_qvalue(
    representation: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    config,
    device: torch.device,
    obs_key: str = OBS_NORMED_KEY,
) -> TensorDictSequential:
    """构建单个 Q 网络（SACLoss 内部复制成 num_qvalue_nets 个集成成员）。输入为已归一化观测。"""
    repr_net = _build_representation(representation, obs_shape, config, device)
    feat_dim = repr_net.output_dim

    return TensorDictSequential(
        TensorDictModule(repr_net, in_keys=[obs_key], out_keys=['feat']),
        TensorDictModule(
            QHead(feat_dim, action_dim, list(config.critic_hidden_size), config.activation, device),
            in_keys=['feat', 'action'],
            out_keys=['state_action_value'],
        ),
    )


def make_qvalue_factory(
    representation: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    config,
    device: torch.device,
) -> Callable[[], TensorDictSequential]:
    return lambda: build_qvalue(representation, obs_shape, action_dim, config, device)
