"""Gymnasium DRL environment wrapping the rigid PROCS simulator."""

import math
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from .procs_core import (
    DT_BASE,
    N_SKIP_SPANS,
    WEAR_AMPLITUDE,
    WEAR_WAVELENGTH,
    ProcsCore,
    actuator_force,
    rigid_overhead_contact_system_params,
    target_contact_force,
)

# 正常回合的折扣回报大约 -10～-20。发散须明显更差。
DIVERGENCE_PENALTY = 1500.0
# 慢均值时间常数。100 Hz 时每步混入约 2.47%，跨距起伏留下大约 4%。
SLOW_MEAN_TAU_S = 0.4
# 二次段到目标力的 5%。再往外改线性，大直流仍有梯度，但不会平方到几千。
SLOW_MEAN_REL_TOL = 0.05


def slow_mean_penalty(relative_bias: float, tol: float = SLOW_MEAN_REL_TOL) -> float:
    """|b|≤tol 为 (b/tol)^2，之外为 2|b|/tol−1。结点上函数值和斜率都连续。"""
    magnitude = abs(relative_bias) / tol
    if magnitude <= 1.0:
        return magnitude**2
    return 2.0 * magnitude - 1.0


class ProcsEnv(gym.Env):
    metadata: ClassVar[dict[str, Any]] = {'render_modes': []}

    def __init__(
        self,
        pantograph: int = 1,
        rigid_overhead_contact_system: int = 3,
        NM: int = 200,
        N_spans: int = 30,
        dt_base: float = DT_BASE,
        speed_kmh: float = 200.0,
        speed_range_kmh: tuple[float, float] | None = (200.0, 300.0),
        irregularity: bool = False,
        wear_amplitude: float = WEAR_AMPLITUDE,
        wear_wavelength: float = WEAR_WAVELENGTH,
        actuator_delta: float = 1.0,
        actuator_noise_std: float = 0.0,
        pantograph_param_scale: dict[str, float] | None = None,
        skip_spans: int = N_SKIP_SPANS,
        control_freq: float = 100.0,
        action_max: float = 100.0,
        action_rate_weight: float = 0.01,
        action_penalty_weight: float = 0.001,
        max_newton_iter: int = 10,
        record_eval_trajectory: bool = False,
    ):
        """speed_range_kmh 给 (lo, hi) 即每回合均匀随机。单个速度训练时把它设为 None。"""
        super().__init__()

        self.pantograph = pantograph
        self.rigid_overhead_contact_system = rigid_overhead_contact_system
        self.NM = NM
        self.N_spans = N_spans
        self.dt_base = dt_base
        self.speed_kmh = speed_kmh
        self.speed_range_kmh = speed_range_kmh
        self.irregularity = bool(irregularity)
        self.wear_amplitude = float(wear_amplitude)
        self.wear_wavelength = float(wear_wavelength)
        self.actuator_delta = float(actuator_delta)
        self.actuator_noise_std = float(actuator_noise_std)
        self.pantograph_param_scale = pantograph_param_scale
        self.skip_spans = skip_spans
        self.action_max = float(action_max)
        self.action_rate_weight = float(action_rate_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.max_newton_iter = max_newton_iter
        self.record_eval_trajectory = bool(record_eval_trajectory)

        if not np.isfinite(dt_base) or dt_base <= 0:
            raise ValueError(f'dt_base must be finite and positive, got {dt_base}')
        if not np.isfinite(speed_kmh) or speed_kmh <= 0:
            raise ValueError(f'speed_kmh must be finite and positive, got {speed_kmh}')
        if not np.isfinite(control_freq) or control_freq <= 0:
            raise ValueError(f'control_freq must be finite and positive, got {control_freq}')
        # 一个控制周期内保持同一动作；周期必须是积分步长的整数倍
        period = 1.0 / float(control_freq)
        n_inner = int(round(period / float(dt_base)))
        if n_inner < 1 or not np.isclose(n_inner * float(dt_base), period, rtol=1e-6, atol=0.0):
            raise ValueError(
                f'1/control_freq must be a positive integer multiple of dt_base, got control_freq={control_freq}, dt_base={dt_base}'
            )
        if not np.isfinite(self.action_max) or self.action_max <= 0:
            raise ValueError(f'action_max must be finite and positive, got {action_max}')
        if not np.isfinite(self.action_rate_weight) or self.action_rate_weight < 0:
            raise ValueError(f'action_rate_weight must be finite and non-negative, got {action_rate_weight}')
        if not np.isfinite(self.action_penalty_weight) or self.action_penalty_weight < 0:
            raise ValueError(f'action_penalty_weight must be finite and non-negative, got {action_penalty_weight}')
        if not np.isfinite(self.wear_amplitude) or self.wear_amplitude < 0:
            raise ValueError(f'wear_amplitude must be finite and non-negative, got {wear_amplitude}')
        if not np.isfinite(self.wear_wavelength) or self.wear_wavelength <= 0:
            raise ValueError(f'wear_wavelength must be finite and positive, got {wear_wavelength}')
        if not np.isfinite(self.actuator_delta) or self.actuator_delta < 0:
            raise ValueError(f'actuator_delta must be finite and non-negative, got {actuator_delta}')
        if not np.isfinite(self.actuator_noise_std) or self.actuator_noise_std < 0:
            raise ValueError(f'actuator_noise_std must be finite and non-negative, got {actuator_noise_std}')
        self._validate_range('speed_range_kmh', speed_range_kmh, lower=0.0, inclusive=False)

        self.control_freq = float(control_freq)
        self.n_inner = n_inner
        self._slow_beta = 1.0 - math.exp(-(1.0 / self.control_freq) / SLOW_MEAN_TAU_S)

        # 可观测：y1[m], v1[m/s], a1[m/s²], v_train[m/s], a_{t-1}。接触力不进观测。
        obs_low = np.array([-1.0, -50.0, -500.0, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0, 50.0, 500.0, 120.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 动作空间：归一化后 [-1, 1]，再映射为 ±action_max N
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        # 最慢车速，用于估算需要最多的控制步
        slowest_speed = speed_range_kmh[0] if speed_range_kmh is not None else speed_kmh
        self.max_episode_steps = self._control_steps_at_speed(float(slowest_speed))

        self._core: ProcsCore | None = None
        self._episode_steps = 0
        self._episode_return = 0.0
        self._previous_action = 0.0

    @staticmethod
    def target_contact_force(speed_kmh: float) -> float:
        return target_contact_force(speed_kmh)

    @staticmethod
    def _validate_range(
        name: str,
        value: tuple[float, float] | None,
        *,
        lower: float,
        inclusive: bool = True,  # 决定下界是大于等于还是严格大于
    ) -> None:
        if value is None:
            return
        try:
            lo, hi = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a (lo, hi) pair or None') from exc
        try:
            invalid = (
                not np.isfinite(lo) or not np.isfinite(hi) or lo < lower or (not inclusive and lo <= lower) or lo > hi
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must contain two finite numbers') from exc
        if invalid:
            bound = f'>= {lower}' if inclusive else f'> {lower}'
            raise ValueError(f'{name} must satisfy finite lo {bound} and lo <= hi, got {value}')

    def _sample_speed(self) -> float:
        if self.speed_range_kmh is None:
            return float(self.speed_kmh)
        lo, hi = self.speed_range_kmh
        return float(self.np_random.uniform(lo, hi))

    def _control_steps_at_speed(self, speed_kmh: float) -> int:
        """Return the episode length at a fixed speed without running the simulator."""
        span_length, n_spans, *_ = rigid_overhead_contact_system_params(
            self.rigid_overhead_contact_system,
            self.N_spans,
        )
        velocity = speed_kmh / 3.6
        total_length = span_length * n_spans
        n_steps = int(np.floor(total_length / velocity / self.dt_base)) + 1
        positions = velocity * (np.arange(n_steps) * self.dt_base)
        k_start = int(np.searchsorted(positions, self.skip_spans * span_length, side='left'))
        k_end = int(np.searchsorted(positions, total_length - self.skip_spans * span_length, side='right'))
        return math.ceil((k_end - max(1, k_start)) / self.n_inner)

    def _observation(self) -> np.ndarray:
        # [y1, v1, a1, v_train, a_{t-1}]。Fc 留在仿真量里。
        plant = self._core.current_observation().astype(np.float32)
        return np.concatenate([plant[:4], np.array([self._previous_action], dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        speed = self._sample_speed()

        if self._core is None:
            self._core = ProcsCore(
                rigid_overhead_contact_system=self.rigid_overhead_contact_system,
                pantograph=self.pantograph,
                speed_kmh=speed,
                NM=self.NM,
                N_spans=self.N_spans,
                dt_base=self.dt_base,
                irregularity=self.irregularity,
                wear_amplitude=self.wear_amplitude,
                wear_wavelength=self.wear_wavelength,
                pantograph_param_scale=self.pantograph_param_scale,
                max_newton_iter=self.max_newton_iter,
                skip_spans=self.skip_spans,
            )
        else:
            self._core.reset(speed_kmh=speed)

        # 无控制滑行至稳定段起点，启动瞬态不计入训练。慢均值从这段原始接触力起步。
        burn = self._core.k_start - 1
        coast_fc = np.empty(0)
        if burn > 0:
            coast_fc = self._core.advance(burn, 0.0)['contact_force']
        self._slow_mean = float(coast_fc.mean()) if coast_fc.size else float(self._core.fc)

        self._episode_steps = 0
        self._episode_return = 0.0
        self._previous_action = 0.0

        obs = self._observation()
        remaining = self._core.k_end - self._core.k
        info = {
            'F0_N': float(self._core.F0),
            'speed_kmh': float(self._core.speed_kmh),
            'target_Fc_N': self.target_contact_force(self._core.speed_kmh),
            'n_control_steps': math.ceil(remaining / self.n_inner),
            'n_inner': int(self.n_inner),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._core is None:
            raise RuntimeError('step() called before reset()')

        action = np.asarray(action, dtype=np.float32)
        if action.size != 1 or not np.isfinite(action).all():
            raise ValueError('action must contain exactly one finite value')
        normalized_action = float(np.clip(action.reshape(-1)[0], -1.0, 1.0))
        f3 = normalized_action * self.action_max
        noise = 0.0 if self.actuator_noise_std == 0.0 else float(self.np_random.normal(0.0, self.actuator_noise_std))
        # 每个控制步抽一次噪声，并在该控制周期内保持，和 f3 的零阶保持一致。
        u = actuator_force(f3, self.actuator_delta, noise)

        try:
            hist = self._core.advance(self.n_inner, u)
        except FloatingPointError:
            self._previous_action = normalized_action
            obs = self._observation()
            if not np.isfinite(obs).all():
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._episode_steps += 1
            self._episode_return += -DIVERGENCE_PENALTY
            return (
                obs,
                -DIVERGENCE_PENALTY,
                True,
                False,
                {
                    'diverged': True,
                    'reason': 'contact_nonconvergence',
                    'episode_step': self._episode_steps,
                    'episode_return': self._episode_return,
                },
            )
        fc = hist['contact_force']  # 控制步内的接触力
        if hist['n_actual'] == 0:
            # 这一步没有推进仿真，观测仍对应当前状态和产生它的动作
            return self._observation(), 0.0, False, True, {}

        # 发散 / NaN：仿真状态已失效，统一终止并给大惩罚（离线 fc=0 不算发散）
        plant_ok = np.isfinite(self._core.current_observation()).all()
        if (fc.size and not np.isfinite(fc).all()) or not plant_ok:
            self._previous_action = normalized_action
            obs = self._observation()
            if not plant_ok:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._episode_steps += 1
            self._episode_return += -DIVERGENCE_PENALTY
            return (
                obs,
                -DIVERGENCE_PENALTY,
                True,
                False,
                {
                    'diverged': True,
                    'reason': 'contact_nonconvergence',
                    'episode_step': self._episode_steps,
                    'episode_return': self._episode_return,
                },
            )

        target_force = self.target_contact_force(self._core.speed_kmh)
        n_actual = int(hist['n_actual'])
        mean_fc = float(fc.mean())
        std_fc = float(fc.std())
        # 原始接触力。e_t 同时含这一步里的波动和偏差。
        track_error = float(np.mean(((fc - target_force) / target_force) ** 2))
        n_loss = int((fc <= 0.0).sum())  # 离线步数，只进统计
        loss_fraction = n_loss / n_actual
        # 单边接触把力截断在 0。分离深度用弹簧力 KS*间隙，只在为负时按相对深度平方扣。
        signed = hist['contact_force_signed']
        loss_penalty = float(np.mean((np.minimum(signed, 0.0) / target_force) ** 2))
        min_fc = float(fc.min())
        max_fc = float(fc.max())
        action_penalty = self.action_penalty_weight * normalized_action**2  # 惩罚动作幅度大
        action_delta = normalized_action - self._previous_action
        action_rate_penalty = self.action_rate_weight * action_delta**2  # 惩罚动作变化太快
        self._slow_mean = (1.0 - self._slow_beta) * self._slow_mean + self._slow_beta * mean_fc
        slow_penalty = slow_mean_penalty((self._slow_mean - target_force) / target_force)

        reward = -(track_error + slow_penalty + loss_penalty + action_penalty + action_rate_penalty)

        self._previous_action = normalized_action
        obs = self._observation()

        # 稳定段跑完是时间性截断：TD 目标仍自举末状态价值；terminated 留给发散
        terminated = False
        truncated = self._core.done()

        self._episode_steps += 1
        self._episode_return += reward

        info = {
            'mean_Fc_N': mean_fc,
            'std_Fc_N': std_fc,
            'min_Fc_N': min_fc,
            'max_Fc_N': max_fc,
            'force_mse': track_error,
            'track_error': track_error,
            'loss_fraction': loss_fraction,
            'loss_penalty': loss_penalty,
            'slow_mean_N': self._slow_mean,
            'slow_penalty': slow_penalty,
            'signed_Fc_N': signed.astype(np.float64, copy=False),
            'n_loss': n_loss,
            'f3_N': f3,
            'u_N': u,
            'n_inner_actual': n_actual,
            'target_Fc_N': target_force,
            'action_delta': action_delta,
            'action_rate_penalty': action_rate_penalty,
            'episode_step': self._episode_steps,
            'episode_return': self._episode_return,
        }
        if self.record_eval_trajectory:
            info['trajectory_time_s'] = hist['time_s']
            info['trajectory_position_m'] = hist['position_m']
            info['trajectory_contact_force_N'] = fc.astype(np.float64, copy=False)
            info['trajectory_control_force_N'] = np.full(n_actual, u, dtype=np.float64)
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def parse_history_length(value, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f'history_length must be an integer, got bool {value!r}')
    if type(value) is not int:
        raise TypeError(f'history_length must be a non-negative integer or None, got {type(value).__name__}: {value!r}')
    if value < 0:
        raise ValueError(f'history_length must be non-negative, got {value}')
    return value
