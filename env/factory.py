"""TorchRL environment factory for PROCS (train collectors) — eval still uses plain Gymnasium envs."""

from __future__ import annotations

import copy
import functools

import torch
from torchrl.envs import CatFrames, Compose, GymWrapper, SqueezeTransform, TransformedEnv, UnsqueezeTransform

from .procs_env import ProcsEnv, parse_history_length


def make_procs_env(env_kwargs: dict, *, history_length: int | None = None) -> TransformedEnv:
    """Wrap the Gymnasium PROCS env as a TorchRL TransformedEnv.

    history_length > 0 时用 CatFrames 把最近 N 帧观测沿新维度堆叠成 (N, obs_dim)，
    padding 语义与 gymnasium FrameStackObservation(padding_type='reset') 一致。
    """
    kwargs = dict(env_kwargs)
    if history_length is None:
        history_length = parse_history_length(kwargs.get('history_length'), default=0)
    kwargs.pop('history_length', None)
    base_env = GymWrapper(ProcsEnv(**kwargs))

    transforms = []
    if history_length > 0:
        # (obs_dim,) -> (1, 1, obs_dim) -> 沿第 0 维堆叠 (N, 1, obs_dim) -> (N, obs_dim)
        transforms += [
            UnsqueezeTransform(dim=-2, in_keys=['observation'], out_keys=['observation']),
            UnsqueezeTransform(dim=-3, in_keys=['observation'], out_keys=['observation']),
            CatFrames(dim=-3, N=history_length, in_keys=['observation']),
            SqueezeTransform(dim=-2, in_keys=['observation'], out_keys=['observation']),
        ]
    return TransformedEnv(base_env, Compose(*transforms))


def _worker_env(index: int, env_kwargs: dict, history_length: int, seed: int | None) -> TransformedEnv:
    """ParallelEnv worker 工厂（模块级函数，保证 spawn 下可 pickle）。"""
    env = make_procs_env(env_kwargs, history_length=history_length)
    if seed is not None:
        env.set_seed(int(seed) + index)
    return env


def close_envs(envs) -> None:
    """安全关闭 TorchRL 向量环境（未启动过的环境 close() 会抛 RuntimeError）。"""
    try:
        envs.close()
    except RuntimeError:
        pass


def make_envs(config):
    """Build the vectorized TorchRL env used by the collector.

    config.vectorize: 'SubprocVecEnv' -> ParallelEnv, 'DummyVecEnv' -> SerialEnv.
    config.env_seed  : 播种第 0 个环境，后续环境依次 +1。
    """
    from torchrl.envs import ParallelEnv, SerialEnv

    env_kwargs = copy.deepcopy(getattr(config, 'env_kwargs', {}) or {})
    history_length = parse_history_length(env_kwargs.get('history_length'), default=0)
    num_envs = int(getattr(config, 'parallels', 1))
    vectorize = getattr(config, 'vectorize', 'SubprocVecEnv')
    seed = getattr(config, 'env_seed', None)

    env_fns = [
        functools.partial(_worker_env, i, env_kwargs, history_length, seed) for i in range(num_envs)
    ]
    env_cls = ParallelEnv if vectorize == 'SubprocVecEnv' else SerialEnv
    envs = env_cls(num_envs, env_fns)
    torch.manual_seed(int(getattr(config, 'seed', 0) or 0))
    return envs
