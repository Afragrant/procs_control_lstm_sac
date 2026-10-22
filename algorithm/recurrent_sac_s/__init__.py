"""Recurrent SAC：演员与批评者共享一条 LSTM，只有演员更新它。"""

from .agent import RecurrentSACSAgent

__all__ = ['RecurrentSACSAgent']
