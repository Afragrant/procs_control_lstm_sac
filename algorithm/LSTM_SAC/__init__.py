"""LSTM-SAC representation package (plain torch, no framework registry)."""

from .representation import LSTM_HIDDEN, HistoryLSTM

__all__ = ['LSTM_HIDDEN', 'HistoryLSTM']
