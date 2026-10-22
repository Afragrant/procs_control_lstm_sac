"""Recurrent Greedy SAC：独立 LSTM 记忆，actor 用 proposal 的精英动作做最大似然。"""

from .agent import RecurrentGreedySACAgent

__all__ = ['RecurrentGreedySACAgent']
