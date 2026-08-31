"""Discrete PID controller."""

from __future__ import annotations

import math


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float, output_limit: float):
        values = (kp, ki, kd, dt, output_limit)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('PID parameters must be finite')
        if dt <= 0 or output_limit <= 0:
            raise ValueError('dt and output_limit must be positive')
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt, self.output_limit = dt, output_limit
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0  # 此前接受的误差积分
        self.previous_error: float | None = None  # 上一个采样周期的误差

    def update(self, setpoint: float, measurement: float) -> float:
        if not math.isfinite(setpoint) or not math.isfinite(measurement):
            raise ValueError('setpoint and measurement must be finite')
        error = setpoint - measurement  # 误差
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / self.dt  # 微分项
        candidate_integral = self.integral + error * self.dt  # 积分项
        output = self.kp * error + self.ki * candidate_integral + self.kd * derivative  # PID 输出

        # 未超过限制时，更新积分项
        if abs(output) <= self.output_limit:
            self.integral = candidate_integral
        self.previous_error = error
        return max(-self.output_limit, min(output, self.output_limit))
