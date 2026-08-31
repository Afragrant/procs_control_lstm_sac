"""Shared train/eval helpers for SAC and LSTM-SAC run scripts."""

from __future__ import annotations

import csv
import math
import os
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation

from algorithm.PID import PIDController
from algorithm.SAC import SACAgent
from env.procs_env import ProcsEnv, parse_history_length

FINAL_MODEL_NAME = 'final_train_model.pth'
BEST_MODEL_DIR = 'best'
BEST_MODEL_NAME = 'best_model.pth'
# 训练范围设为 speed_range_kmh 时（或测试不带 --speed 时）使用的验证速度点
EVAL_SPEEDS_KMH = (200.0, 230.0, 250.0)
EVAL_SEED = 1

# 轨迹 CSV 列定义
TRAJECTORY_FIELDS = (
    'speed_kmh',
    'policy',
    'sample_index',
    'time_s',
    'position_m',
    'contact_force_N',
    'control_force_N',
    'target_force_N',
)

# 波形图策略样式
POLICY_PLOT_STYLES = {
    'mlp_sac': {'label': 'MLP-SAC', 'color': 'C0'},
    'lstm_sac': {'label': 'LSTM-SAC', 'color': 'C1'},
    'pid': {'label': 'PID', 'color': 'C3'},
    'zero_control': {'label': 'zero control', 'color': 'C2'},
}


# 基于全部内部采样的四个核心统计量
def compute_contact_force_sample_stats(contact_force: np.ndarray) -> dict[str, float | int]:
    fc = np.asarray(contact_force, dtype=np.float64)
    if fc.size == 0:
        return {
            'fc_mean_N': float('nan'),
            'fc_min_N': float('nan'),
            'fc_max_N': float('nan'),
            'fc_sample_std_N': float('nan'),
            'n_samples': 0,
        }
    return {
        'fc_mean_N': float(fc.mean()),
        'fc_min_N': float(fc.min()),
        'fc_max_N': float(fc.max()),
        'fc_sample_std_N': float(fc.std(ddof=1)) if fc.size > 1 else 0.0,
        'n_samples': int(fc.size),
    }


# 把每步高频样本拼成整条稳定段波形
def _append_trajectory_samples(step_info: dict, trajectory: dict[str, list]) -> int:
    contact_force = step_info.get('trajectory_contact_force_N')
    if contact_force is None:
        return 0
    n_samples = int(np.asarray(contact_force).size)
    if n_samples == 0:
        return 0
    trajectory['time_s'].extend(np.asarray(step_info['trajectory_time_s'], dtype=np.float64).tolist())
    trajectory['position_m'].extend(np.asarray(step_info['trajectory_position_m'], dtype=np.float64).tolist())
    trajectory['contact_force_N'].extend(np.asarray(contact_force, dtype=np.float64).tolist())
    trajectory['control_force_N'].extend(np.asarray(step_info['trajectory_control_force_N'], dtype=np.float64).tolist())
    return n_samples


# SAC 测试必须给 --model。
def validate_sac_test_model(command: str, model: str | None) -> None:
    if command == 'test' and not model:
        raise ValueError('test requires --model <checkpoint>')


# 堆叠观测 SAC 测试必须给 --model。
def validate_lstm_sac_test_model(command: str, model: str | None, *, label: str = 'lstm-sac') -> None:
    if command == 'test' and not model:
        raise ValueError(f'test requires --model <{label}-checkpoint>')


# 只在 train 时检查，保证训练真的能更新一次网络
def validate_sac_training_steps(
    command: str,
    running_steps: int,
    parallels: int,
    *,
    start_training: int = 0,
    training_frequency: int = 1,
) -> None:
    if command == 'test':
        return
    if parallels <= 0:
        raise ValueError(f'parallels must be positive, got {parallels}')
    if training_frequency <= 0:
        raise ValueError(f'training_frequency must be positive, got {training_frequency}')
    if start_training < 0:
        raise ValueError(f'start_training must be non-negative, got {start_training}')
    if running_steps <= 0:
        raise ValueError(f'running_steps must be positive, got {running_steps}')
    if running_steps < parallels:
        raise ValueError(f'running_steps ({running_steps}) must be >= parallels ({parallels})')
    if running_steps % parallels != 0:
        raise ValueError(f'running_steps ({running_steps}) must be divisible by parallels ({parallels})')
    interval = math.lcm(parallels, training_frequency)
    first_update_step = (start_training // interval + 1) * interval
    if running_steps <= first_update_step:
        raise ValueError(
            f'running_steps ({running_steps}) must be > first possible update step ({first_update_step}) '
            f'(start_training={start_training}, interval={interval})'
        )


def validate_device(device) -> None:
    requested = torch.device(str(device))
    if requested.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'{requested} was requested, but CUDA is not available')


def print_device(agent: SACAgent) -> None:
    param_device = next(agent.policy.parameters()).device
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'config.device: {agent.config.device}')
    print(f'policy device: {param_device}')
    print('environment workers: CPU (NumPy/SciPy)')


def assert_spaces(envs, *, obs_shape: tuple[int, ...]) -> None:
    actual_obs = tuple(envs.observation_spec_unbatched['observation'].shape)
    act_shape = tuple(envs.action_spec_unbatched.shape)
    print(f'vec env observation_spec.shape={actual_obs} action_spec.shape={act_shape}')
    if actual_obs != obs_shape or act_shape != (1,):
        raise RuntimeError(f'expected obs {obs_shape} and action (1,), got {actual_obs} and {act_shape}')


def configure_speed_run(config, speed: float | None = None) -> tuple[dict, str]:
    env_kwargs = dict(getattr(config, 'env_kwargs', {}) or {})
    if speed is not None:
        env_kwargs.update(speed_kmh=float(speed), speed_range_kmh=None)

    speed_range = env_kwargs.get('speed_range_kmh')
    if speed_range is None:
        train_speed = float(env_kwargs.get('speed_kmh', 250.0))
        if not math.isfinite(train_speed) or train_speed <= 0:
            raise ValueError(f'speed must be a positive finite number, got {train_speed}')
        env_kwargs['speed_kmh'] = train_speed
        speed_dir = f'{train_speed:g}kmh'
    else:
        if len(speed_range) != 2:
            raise ValueError(f'speed_range_kmh must contain two values, got {speed_range!r}')
        low, high = map(float, speed_range)
        if not all(math.isfinite(value) and value > 0 for value in (low, high)) or low > high:
            raise ValueError(f'invalid speed_range_kmh: {speed_range!r}')
        env_kwargs['speed_range_kmh'] = [low, high]
        speed_dir = f'{low:g}-{high:g}kmh'

    config.env_kwargs = env_kwargs
    config.log_dir = str(Path(config.log_dir) / speed_dir)
    config.model_dir = str(Path(config.model_dir) / speed_dir)
    return env_kwargs, speed_dir


def configure_irregularity(
    env_kwargs: dict,
    *,
    irregularity: bool | None = None,
    wear_amplitude: float | None = None,
    wear_wavelength: float | None = None,
) -> dict:
    kwargs = dict(env_kwargs)
    if irregularity is not None:
        kwargs['irregularity'] = irregularity
    if wear_amplitude is not None:
        kwargs['wear_amplitude'] = wear_amplitude
    if wear_wavelength is not None:
        kwargs['wear_wavelength'] = wear_wavelength
    return kwargs


def zero_obs(obs_shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros((1, *obs_shape), dtype=np.float32)


def finite_action_check(agent: SACAgent) -> None:
    obs_shape = agent.obs_shape
    obs = zero_obs(obs_shape)
    a1 = agent.get_actions(obs, test_mode=True)['actions']
    a2 = agent.get_actions(obs, test_mode=True)['actions']
    if not np.isfinite(a1).all():
        raise RuntimeError(f'non-finite deterministic action: {a1}')
    if not np.array_equal(a1, a2):
        raise RuntimeError('deterministic actions differ for the same observation')
    print(f'deterministic action check ok (obs={obs_shape}): {a1.reshape(-1)}')


# 评估环境构建
def make_eval_env(env_kwargs: dict, *, history_length: int | None = None, record_eval_trajectory: bool = True):
    kwargs = {k: v for k, v in env_kwargs.items() if k != 'history_length'}
    if history_length is None:
        # 未显式传参时回退到 env_kwargs，避免调用点漏传导致观测维度不匹配
        history_length = parse_history_length(env_kwargs.get('history_length'), default=0)
    env = ProcsEnv(**kwargs, record_eval_trajectory=record_eval_trajectory)
    if history_length:
        env = FrameStackObservation(env, stack_size=history_length, padding_type='reset')
    return env


def make_periodic_eval_env(env_kwargs: dict, *, history_length: int | None = None):
    return make_eval_env(env_kwargs, history_length=history_length, record_eval_trajectory=True)


# 评估用的单回合 rollout 函数
def run_episode(env, action_fn, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    rows = []
    trajectory = {
        'time_s': [],
        'position_m': [],
        'contact_force_N': [],
        'control_force_N': [],
    }
    n_inner_total = n_loss_total = n_over_total = 0
    sum_mse = 0.0
    rewards = []
    f3s = []
    diverged = False
    reason = ''
    while True:
        action = np.asarray(action_fn(obs), dtype=np.float32).reshape(1)
        obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(float(reward))
        f3s.append(float(step_info.get('f3_N', 0.0)))
        if step_info.get('diverged', False):
            diverged = True
            reason = str(step_info.get('reason', 'diverged'))
            break
        n_inner = int(step_info.get('n_inner_actual', 0))
        if n_inner > 0:
            _append_trajectory_samples(step_info, trajectory)
            n_inner_total += n_inner
            n_loss_total += int(step_info['n_loss'])
            n_over_total += int(step_info['n_overforce'])
            sum_mse += float(step_info['force_mse']) * n_inner
            rows.append(
                {
                    'step': int(step_info['episode_step']),
                    'speed_kmh': float(info['speed_kmh']),
                    'Fc_N': float(step_info['mean_Fc_N']),
                    'f3_N': float(step_info['f3_N']),
                    'force_mse': float(step_info['force_mse']),
                    'loss_fraction': float(step_info['loss_fraction']),
                    'overforce_fraction': float(step_info['overforce_fraction']),
                    'reward': float(reward),
                    'episode_return': float(step_info['episode_return']),
                }
            )
        if terminated or truncated:
            break
    sample_stats = compute_contact_force_sample_stats(np.asarray(trajectory['contact_force_N'], dtype=np.float64))
    force_mse = float(sum_mse / n_inner_total) if n_inner_total else float('nan')
    target_force = float(info['target_Fc_N'])
    fc_mean = float(sample_stats['fc_mean_N'])
    return {
        'speed_kmh': float(info['speed_kmh']),
        'target_Fc_N': target_force,
        'episode_return': float(sum(rewards)),
        'fc_mean_N': fc_mean,
        'fc_bias_N': fc_mean - target_force,
        'fc_min_N': float(sample_stats['fc_min_N']),
        'fc_max_N': float(sample_stats['fc_max_N']),
        'fc_sample_std_N': float(sample_stats['fc_sample_std_N']),
        'fc_rmse_N': target_force * math.sqrt(force_mse),
        'force_mse': force_mse,
        'loss_fraction': float(n_loss_total / n_inner_total) if n_inner_total else float('nan'),
        'overforce_fraction': float(n_over_total / n_inner_total) if n_inner_total else float('nan'),
        'f3_rms_N': float(np.sqrt(np.mean(np.square(f3s)))) if f3s else 0.0,
        'n_control_steps': len(rows),
        'n_samples': int(sample_stats['n_samples']),
        'diverged': diverged,
        'reason': reason,
        'rows': rows,
        'trajectory': trajectory,
    }


def agent_action_fn(agent: SACAgent):
    obs_shape = agent.obs_shape

    def _fn(obs):
        batched = np.asarray(obs, dtype=np.float32)
        if batched.ndim == len(obs_shape):
            batched = batched[None, ...]
        return agent.get_actions(batched, test_mode=True)['actions'][0]

    return _fn


# 相对改善百分比
def pct_improve(baseline: float, value: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return float('nan')
    return (baseline - value) / abs(baseline) * 100.0


# 逐步轨迹写 CSV
def episode_status(result: dict) -> str:
    if result.get('diverged'):
        reason = str(result.get('reason') or 'unknown')
        return f'diverged:{reason}'
    return 'ok'


def print_eval_table(summaries: list[dict], policies: tuple[str, ...]) -> None:
    print(
        f'{"speed":>8} {"policy":>12} {"status":>24} {"target":>10} '
        f'{"mean":>10} {"bias":>10} {"rmse":>10} {"std":>10} {"n_samp":>8}'
    )
    for speed in sorted({float(row['speed_kmh']) for row in summaries}):
        pair = {row['policy']: row for row in summaries if row['speed_kmh'] == speed}
        for policy in policies:
            row = pair[policy]
            print(
                f'{speed:8.0f} {policy:>12} {episode_status(row):>24} '
                f'{row["target_Fc_N"]:10.4f} {row["fc_mean_N"]:10.4f} '
                f'{row["fc_bias_N"]:10.4f} {row["fc_rmse_N"]:10.4f} '
                f'{row["fc_sample_std_N"]:10.4f} {row["n_samples"]:8d}'
            )


def _eval_speed_kwargs(env_kwargs: dict, speed: float) -> dict:
    kwargs = dict(env_kwargs)
    kwargs['speed_kmh'] = speed
    kwargs['speed_range_kmh'] = None
    return kwargs


def eval_condition_name(env_kwargs: dict) -> str:
    if not env_kwargs.get('irregularity', False):
        return 'smooth'
    return f'irregularity_A{float(env_kwargs["wear_amplitude"]):g}m_L{float(env_kwargs["wear_wavelength"]):g}m'


def _validation_speeds(env_kwargs: dict) -> tuple[float, ...]:
    if env_kwargs.get('speed_range_kmh') is None and 'speed_kmh' in env_kwargs:
        return (float(env_kwargs['speed_kmh']),)
    return EVAL_SPEEDS_KMH


def write_trajectory_csv(
    out_path: Path,
    runs_by_speed: dict[float, dict[str, dict]],
    policies: tuple[str, ...],
) -> None:
    with out_path.open('w', newline='') as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=TRAJECTORY_FIELDS)
        writer_csv.writeheader()
        for speed in runs_by_speed:
            for policy in policies:
                result = runs_by_speed[speed][policy]
                trajectory = result['trajectory']
                target_force = float(result['target_Fc_N'])
                for sample_index, (
                    time_s,
                    position_m,
                    contact_force_N,
                    control_force_N,
                ) in enumerate(
                    zip(
                        trajectory['time_s'],
                        trajectory['position_m'],
                        trajectory['contact_force_N'],
                        trajectory['control_force_N'],
                        strict=True,
                    )
                ):
                    writer_csv.writerow(
                        {
                            'speed_kmh': speed,
                            'policy': policy,
                            'sample_index': sample_index,
                            'time_s': time_s,
                            'position_m': position_m,
                            'contact_force_N': contact_force_N,
                            'control_force_N': control_force_N,
                            'target_force_N': target_force,
                        }
                    )


def plot_contact_force_comparison(
    out_path: Path,
    runs_by_speed: dict[float, dict[str, dict]],
    policies: tuple[str, ...],
) -> None:
    speeds = tuple(runs_by_speed)
    fig, axes = plt.subplots(len(speeds), 1, figsize=(10, 3 * len(speeds)), sharex=False)
    if len(speeds) == 1:
        axes = [axes]
    for axis, speed in zip(axes, speeds, strict=True):
        target_force = None
        for policy in policies:
            result = runs_by_speed[speed][policy]
            trajectory = result['trajectory']
            if not trajectory['position_m']:
                continue
            style = POLICY_PLOT_STYLES.get(policy, {'label': policy, 'color': None})
            axis.plot(
                trajectory['position_m'],
                trajectory['contact_force_N'],
                label=style['label'],
                color=style['color'],
                linewidth=0.8,
            )
            target_force = float(result['target_Fc_N'])
        if target_force is not None:
            axis.axhline(target_force, color='k', linestyle='--', linewidth=1.0, label='target')
        axis.set_title(f'{int(speed)} km/h')
        axis.set_xlabel('position_m')
        axis.set_ylabel('contact_force_N')
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_eval_csv(
    out_dir: Path,
    runs_by_speed: dict[float, dict[str, dict]],
    policies: tuple[str, ...],
    writer=None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'control_compare.csv').unlink(missing_ok=True)
    summaries = [
        {
            'policy': policy,
            **{k: v for k, v in runs_by_speed[speed][policy].items() if k not in ('rows', 'trajectory')},
        }
        for speed in runs_by_speed
        for policy in policies
    ]
    if writer is not None:
        for row in summaries:
            writer.add_scalar(
                f'eval/{int(row["speed_kmh"])}/{row["policy"]}/contact_force_std_N',
                row['fc_sample_std_N'],
                0,
            )
            writer.add_scalar(
                f'eval/{int(row["speed_kmh"])}/{row["policy"]}/contact_force_rmse_N',
                row['fc_rmse_N'],
                0,
            )
    trajectory_path = out_dir / 'contact_force_trajectory.csv'
    write_trajectory_csv(trajectory_path, runs_by_speed, policies)
    print(f'wrote {trajectory_path}')
    plot_path = out_dir / 'contact_force_comparison.png'
    plot_contact_force_comparison(plot_path, runs_by_speed, policies)
    print(f'wrote {plot_path}')
    summary_path = out_dir / 'control_summary.csv'
    summary_fields = [
        'speed_kmh',
        'policy',
        'status',
        'target_Fc_N',
        'fc_mean_N',
        'fc_bias_N',
        'fc_rmse_N',
        'fc_sample_std_N',
        'n_samples',
    ]
    with summary_path.open('w', newline='') as fh:
        writer_summary = csv.DictWriter(fh, fieldnames=summary_fields)
        writer_summary.writeheader()
        for row in summaries:
            writer_summary.writerow(
                {
                    'speed_kmh': row['speed_kmh'],
                    'policy': row['policy'],
                    'status': episode_status(row),
                    'target_Fc_N': row['target_Fc_N'],
                    'fc_mean_N': row['fc_mean_N'],
                    'fc_bias_N': row['fc_bias_N'],
                    'fc_rmse_N': row['fc_rmse_N'],
                    'fc_sample_std_N': row['fc_sample_std_N'],
                    'n_samples': row['n_samples'],
                }
            )
    print(f'wrote {summary_path}')
    print_eval_table(summaries, policies)
    return summaries


def is_valid_periodic_eval_result(result: dict) -> bool:
    if result.get('diverged', False):
        return False
    if int(result.get('n_samples', 0)) < 2:
        return False
    return bool(np.isfinite(result['fc_rmse_N']))


def compute_best_score(speed_results: dict[float, dict], baseline_results: dict[float, dict]) -> float:
    ratios = [
        max(
            float(result['fc_rmse_N']) / float(baseline_results[speed]['fc_rmse_N']),
            float(result['fc_sample_std_N']) / float(baseline_results[speed]['fc_sample_std_N']),
        )
        for speed, result in speed_results.items()
    ]
    if not all(np.isfinite(ratio) for ratio in ratios):
        return float('nan')
    return float(np.mean(ratios))


def _log_periodic_eval(
    agent: SACAgent,
    speed_results: dict[float, dict],
    baseline_results: dict[float, dict],
    score: float,
    step: int,
) -> None:
    writer = getattr(agent, 'writer', None)
    if writer is None:
        return
    for speed, result in speed_results.items():
        writer.add_scalar(f'periodic_eval/{int(speed)}/contact_force_std_N', result['fc_sample_std_N'], step)
        writer.add_scalar(f'periodic_eval/{int(speed)}/contact_force_rmse_N', result['fc_rmse_N'], step)
        writer.add_scalar(
            f'periodic_eval/{int(speed)}/zero_control_std_N', baseline_results[speed]['fc_sample_std_N'], step
        )
        writer.add_scalar(f'periodic_eval/{int(speed)}/zero_control_rmse_N', baseline_results[speed]['fc_rmse_N'], step)
    if np.isfinite(score):
        writer.add_scalar('periodic_eval/mean_joint_ratio', score, step)


def _log_best_score(agent: SACAgent, score: float, step: int) -> None:
    writer = getattr(agent, 'writer', None)
    if writer is not None and np.isfinite(score):
        writer.add_scalar('periodic_eval/best_joint_ratio', score, step)


def periodic_evaluate(
    agent: SACAgent,
    env_kwargs: dict,
    *,
    history_length: int | None = None,
) -> tuple[dict[float, dict], dict[float, dict], float, bool]:
    speed_results: dict[float, dict] = {}
    baseline_results: dict[float, dict] = {}
    eval_speeds = _validation_speeds(env_kwargs)
    for speed in eval_speeds:
        kwargs = _eval_speed_kwargs(env_kwargs, speed)
        eval_env = make_periodic_eval_env(kwargs, history_length=history_length)
        try:
            speed_results[speed] = run_episode(eval_env, agent_action_fn(agent), seed=EVAL_SEED)
            baseline_results[speed] = run_episode(eval_env, lambda _obs: np.zeros(1, dtype=np.float32), seed=EVAL_SEED)
        finally:
            eval_env.close()

    valid = all(
        is_valid_periodic_eval_result(speed_results[speed]) and is_valid_periodic_eval_result(baseline_results[speed])
        for speed in eval_speeds
    )
    score = compute_best_score(speed_results, baseline_results) if valid else float('nan')
    eligible = valid and all(
        speed_results[speed]['fc_rmse_N'] < baseline_results[speed]['fc_rmse_N']
        and speed_results[speed]['fc_sample_std_N'] < baseline_results[speed]['fc_sample_std_N']
        for speed in eval_speeds
    )
    step = int(getattr(agent, 'current_step', 0))
    _log_periodic_eval(agent, speed_results, baseline_results, score, step)
    return speed_results, baseline_results, score, eligible


def _find_train_info_value(train_info: dict, key: str) -> tuple[bool, float]:
    # 单卡键名为 key；DDP 下 learner 写的是 f'{key}/rank_{rank}'
    if key in train_info:
        return True, train_info[key]
    for name, value in train_info.items():
        if name.startswith(f'{key}/rank_'):
            return True, value
    return False, None


def _segment_had_gradient_update(train_info: dict) -> bool:
    return _find_train_info_value(train_info, 'Qloss')[0] or _find_train_info_value(train_info, 'Ploss')[0]


def _validate_train_info(agent: SACAgent, train_info: dict) -> None:
    if not _segment_had_gradient_update(train_info):
        return
    for key in ('Qloss', 'Ploss'):
        found, value = _find_train_info_value(train_info, key)
        if not found:
            raise RuntimeError(f'missing {key} in train_info after gradient update')
        if not np.isfinite(value):
            raise RuntimeError(f'non-finite {key}: {value}')

    if getattr(agent.config, 'use_automatic_entropy_tuning', False):
        found, alpha = _find_train_info_value(train_info, 'alpha')
        if not found:
            raise RuntimeError('missing alpha in train_info after gradient update')
        if not np.isfinite(alpha):
            raise RuntimeError(f'non-finite alpha: {alpha}')


def _print_periodic_eval(
    step: int,
    speed_results: dict[float, dict],
    baseline_results: dict[float, dict],
    score: float,
    eligible: bool,
) -> None:
    print(f'periodic eval at step {step}:')
    for speed, result in speed_results.items():
        status = episode_status(result)
        baseline = baseline_results[speed]
        print(
            f'  {int(speed):>3} km/h status={status:>24} '
            f'rmse={result["fc_rmse_N"]:.4f} std={result["fc_sample_std_N"]:.4f} '
            f'zero_rmse={baseline["fc_rmse_N"]:.4f} zero_std={baseline["fc_sample_std_N"]:.4f} '
            f'n_samples={result["n_samples"]}'
        )
    print(f'  mean joint ratio={score:.4f} eligible={eligible} (requires RMSE and std below zero control)')


def evaluate_sac(
    agent: SACAgent,
    env_kwargs: dict,
    out_dir: Path,
    *,
    history_length: int | None = None,
    policy_name: str = 'mlp_sac',
    seed: int = EVAL_SEED,
) -> list[dict]:
    """单算法 vs zero-control 评估。

    history_length=None 时回退到 env_kwargs.history_length（与 make_eval_env 一致）；
    MLP-SAC / LSTM-SAC 均为窗口观测。
    """
    runs_by_speed: dict[float, dict[str, dict]] = {}
    for speed in _validation_speeds(env_kwargs):
        kwargs = _eval_speed_kwargs(env_kwargs, speed)
        eval_env = make_eval_env(kwargs, history_length=history_length)
        try:
            runs_by_speed[speed] = {
                policy_name: run_episode(eval_env, agent_action_fn(agent), seed=seed),
                'zero_control': run_episode(eval_env, lambda _obs: np.zeros(1, dtype=np.float32), seed=seed),
            }
        finally:
            eval_env.close()

    summaries = _write_eval_csv(out_dir, runs_by_speed, (policy_name, 'zero_control'), writer=agent.writer)
    overall = {policy_name: [], 'zero_control': []}
    for row in summaries:
        overall[row['policy']].append(row)

    def _mean(key, policy):
        return float(np.mean([row[key] for row in overall[policy]]))

    no_divergence = all(
        not row.get('diverged', False) for row in summaries if row['policy'] in (policy_name, 'zero_control')
    )
    controlled_rmse = _mean('fc_rmse_N', policy_name)
    baseline_rmse = _mean('fc_rmse_N', 'zero_control')
    effective = no_divergence and controlled_rmse < baseline_rmse
    print(
        f'overall rmse {controlled_rmse:.4f} vs {baseline_rmse:.4f} '
        f'({pct_improve(baseline_rmse, controlled_rmse):.2f}% improvement)'
    )
    print(f'effective control: {effective} (criterion: no divergence and lower contact-force RMSE)')
    return summaries


def evaluate_pid(
    env_kwargs: dict,
    out_dir: Path,
    *,
    kp: float,
    ki: float,
    kd: float,
    seed: int = EVAL_SEED,
) -> list[dict]:
    action_max = float(env_kwargs.get('action_max', 100.0))
    control_freq = float(env_kwargs.get('control_freq', 100.0))
    runs_by_speed: dict[float, dict[str, dict]] = {}
    for speed in _validation_speeds(env_kwargs):
        kwargs = _eval_speed_kwargs(env_kwargs, speed)
        eval_env = make_eval_env(kwargs)
        controller = PIDController(kp, ki, kd, 1.0 / control_freq, action_max)
        target = ProcsEnv.target_contact_force(speed)
        try:
            runs_by_speed[speed] = {
                'pid': run_episode(
                    eval_env,
                    lambda obs: np.array([controller.update(target, float(obs[4])) / action_max], dtype=np.float32),
                    seed=seed,
                ),
                'zero_control': run_episode(eval_env, lambda _obs: np.zeros(1, dtype=np.float32), seed=seed),
            }
        finally:
            eval_env.close()
    return _write_eval_csv(out_dir, runs_by_speed, ('pid', 'zero_control'))


def evaluate_lstm_sac(
    lstm_agent: SACAgent,
    sac_agent: SACAgent | None,
    env_kwargs: dict,
    out_dir: Path,
    *,
    history_length: int,
    sac_history_length: int = 0,
    policy_name: str = 'lstm_sac',
    sac_policy_name: str = 'mlp_sac',
    seed: int = EVAL_SEED,
) -> list[dict]:
    """LSTM-SAC vs（可选）MLP-SAC 基线 vs zero-control。两类网络都需要各自的堆叠窗口环境。"""
    policies = (
        (policy_name, sac_policy_name, 'zero_control') if sac_agent is not None else (policy_name, 'zero_control')
    )
    runs_by_speed: dict[float, dict[str, dict]] = {}
    for speed in _validation_speeds(env_kwargs):
        kwargs = _eval_speed_kwargs(env_kwargs, speed)
        with ExitStack() as stack:
            lstm_env = make_eval_env(kwargs, history_length=history_length)
            stack.callback(lstm_env.close)
            runs: dict[str, dict] = {
                policy_name: run_episode(lstm_env, agent_action_fn(lstm_agent), seed=seed),
            }
            if sac_agent is not None:
                sac_env = make_eval_env(kwargs, history_length=sac_history_length)
                stack.callback(sac_env.close)
                runs[sac_policy_name] = run_episode(sac_env, agent_action_fn(sac_agent), seed=seed)
            runs['zero_control'] = run_episode(lstm_env, lambda _obs: np.zeros(1, dtype=np.float32), seed=seed)
            runs_by_speed[speed] = runs

    summaries = _write_eval_csv(
        out_dir,
        runs_by_speed,
        policies,
        writer=lstm_agent.writer,
    )
    short_name = policy_name.removesuffix('_sac') or policy_name
    for speed in runs_by_speed:
        pair = {row['policy']: row for row in summaries if row['speed_kmh'] == speed}
        diverged = [policy for policy in (policy_name, 'zero_control') if pair[policy].get('diverged')]
        if diverged:
            print(f'{speed:8.0f} {f"{short_name} vs zero":>12} comparison invalid (diverged: {", ".join(diverged)})')
            continue
        stacked, base = pair[policy_name], pair['zero_control']
        rmse_p = pct_improve(base['fc_rmse_N'], stacked['fc_rmse_N'])
        std_p = pct_improve(base['fc_sample_std_N'], stacked['fc_sample_std_N'])
        print(
            f'{speed:8.0f} {f"{short_name} vs zero":>12} rmse improvement {rmse_p:10.2f}% (std improvement {std_p:.2f}%)'
        )
    return summaries


def _training_segment_transitions(running_steps: int, trained_transitions: int, eval_interval: int) -> int:
    remaining = running_steps - trained_transitions
    return min(eval_interval, remaining)


def _iter_training_segments(running_steps: int, eval_interval: int) -> list[int]:
    segments: list[int] = []
    trained = 0
    while trained < running_steps:
        segment = _training_segment_transitions(running_steps, trained, eval_interval)
        segments.append(segment)
        trained += segment
    return segments


def _resolve_eval_interval(eval_interval: int | None, agent: SACAgent, running_steps: int) -> int:
    if eval_interval is None:
        resolved = int(getattr(agent.config, 'eval_interval', running_steps))
    else:
        resolved = int(eval_interval)
    if resolved <= 0:
        raise ValueError(f'eval_interval must be positive, got {resolved}')
    return resolved


def _validate_training_schedule(running_steps: int, num_envs: int, eval_interval: int) -> None:
    if running_steps % num_envs != 0:
        raise ValueError(f'running_steps ({running_steps}) must be divisible by num_envs ({num_envs})')
    if eval_interval % num_envs != 0:
        raise ValueError(f'eval_interval ({eval_interval}) must be divisible by num_envs ({num_envs})')
    for segment in _iter_training_segments(running_steps, eval_interval):
        if segment % num_envs != 0:
            raise ValueError(f'training segment size ({segment}) must be divisible by num_envs ({num_envs})')


def _save_best_checkpoint(agent: SACAgent) -> str:
    best_dir = os.path.join(agent.model_dir_save, BEST_MODEL_DIR)
    agent.save_model(BEST_MODEL_NAME, model_path=best_dir)
    return os.path.join(best_dir, BEST_MODEL_NAME)


def train_sac_agent(
    agent: SACAgent,
    *,
    running_steps: int,
    num_envs: int,
    eval_interval: int | None = None,
    env_kwargs: dict | None = None,
    history_length: int | None = None,
) -> None:
    eval_interval = _resolve_eval_interval(eval_interval, agent, running_steps)
    _validate_training_schedule(running_steps, num_envs, eval_interval)

    env_kwargs = dict(env_kwargs or getattr(agent.config, 'env_kwargs', {}) or {})
    segments = _iter_training_segments(running_steps, eval_interval)
    total_vec_steps = running_steps // num_envs
    print(
        f'train {total_vec_steps} vec-steps ({running_steps} transitions, n_envs={num_envs}, '
        f'eval_interval={eval_interval}, segments={segments})'
    )

    best_score = float('inf')
    best_step: int | None = None
    best_per_speed: dict[float, float] = {}
    best_model_path: str | None = None
    any_update = False
    trained_transitions = 0

    for segment_transitions in segments:
        segment_vec_steps = segment_transitions // num_envs
        remaining_transitions = running_steps - trained_transitions
        print(
            f'segment train {segment_vec_steps} vec-steps '
            f'({segment_transitions} transitions, remaining={remaining_transitions})'
        )
        train_info = agent.train(segment_vec_steps)
        if _segment_had_gradient_update(train_info):
            _validate_train_info(agent, train_info)
            any_update = True
        trained_transitions += segment_transitions

        speed_results, baseline_results, score, eligible = periodic_evaluate(
            agent, env_kwargs, history_length=history_length
        )
        _print_periodic_eval(int(agent.current_step), speed_results, baseline_results, score, eligible)
        if any_update and eligible and score < best_score:
            best_score = score
            best_step = int(agent.current_step)
            best_per_speed = {
                speed: max(
                    float(result['fc_rmse_N']) / float(baseline_results[speed]['fc_rmse_N']),
                    float(result['fc_sample_std_N']) / float(baseline_results[speed]['fc_sample_std_N']),
                )
                for speed, result in speed_results.items()
            }
            best_model_path = _save_best_checkpoint(agent)
            _log_best_score(agent, best_score, best_step)
            print(f'new best mean joint ratio {best_score:.4f} at step {best_step}, saved {best_model_path}')

    if not any_update:
        raise RuntimeError('no gradient update occurred during training')

    agent.save_model(FINAL_MODEL_NAME)
    final_model_path = os.path.join(agent.model_dir_save, FINAL_MODEL_NAME)
    print(f'saved {final_model_path}')
    agent.load_model(agent.model_dir_save, FINAL_MODEL_NAME)
    finite_action_check(agent)

    if best_step is not None and best_model_path is not None:
        print(f'best checkpoint step={best_step} mean_joint_ratio={best_score:.4f} path={best_model_path}')
        for speed, ratio in best_per_speed.items():
            print(f'  {int(speed)} km/h worst_rmse_or_std_ratio={ratio:.4f}')
    else:
        print('no eligible best checkpoint saved during training')


def _resolve_seed_dir(seed_dir: Path) -> tuple[str, str]:
    """seed 目录内优先加载 best/best_model.pth，否则回落到 final_train_model.pth。"""
    best_path = seed_dir / BEST_MODEL_DIR / BEST_MODEL_NAME
    if best_path.is_file():
        return str(seed_dir / BEST_MODEL_DIR), BEST_MODEL_NAME
    final_path = seed_dir / FINAL_MODEL_NAME
    if final_path.is_file():
        return str(seed_dir), FINAL_MODEL_NAME
    raise FileNotFoundError(f'no {BEST_MODEL_DIR}/{BEST_MODEL_NAME} or {FINAL_MODEL_NAME} checkpoint in: {seed_dir}')


def load_path(model: str) -> tuple[str, str | None]:
    """Resolve checkpoint path for agent.load_model(load_dir, load_name).

    目录快捷方式优先解析 best/best_model.pth（训练中按周期评估保存的最优模型），
    没有最优模型时回落到 final_train_model.pth；显式给出 .pth 文件路径时按原样加载。
    """
    path = Path(model).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'model path does not exist: {path}')

    if path.is_file():
        if path.suffix.lower() != '.pth':
            raise ValueError(f'expected a .pth checkpoint file, got: {path}')
        return str(path.parent), path.name

    if not path.is_dir():
        raise ValueError(f'model path is not a file or directory: {path}')

    if path.name.startswith('seed_'):
        return _resolve_seed_dir(path)

    seed_children = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith('seed_'))
    if seed_children:
        return _resolve_seed_dir(seed_children[-1])

    final_path = path / FINAL_MODEL_NAME
    if final_path.is_file():
        return str(path), FINAL_MODEL_NAME

    raise ValueError(
        f'unrecognized model path layout: {path} '
        '(expected .pth file, seed_* directory, or model root with seed_* subdirectories)'
    )


def eval_run_name(load_dir: str) -> str:
    path = Path(load_dir)
    return next((parent.name for parent in (path, *path.parents) if parent.name.startswith('seed_')), path.name)
