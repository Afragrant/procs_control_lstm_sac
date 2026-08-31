"""MLP-SAC representation package（历史窗口展平 MLP，纯 torch，无框架注册）。"""

from .representation import DEFAULT_MLP_HIDDEN, HistoryMLP

__all__ = ['DEFAULT_MLP_HIDDEN', 'HistoryMLP']
