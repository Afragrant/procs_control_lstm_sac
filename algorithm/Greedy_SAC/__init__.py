"""Greedy SAC：条件交叉熵更新 actor，熵只留在 proposal 上。"""

from .agent import GreedySACAgent

__all__ = ['GreedySACAgent']
