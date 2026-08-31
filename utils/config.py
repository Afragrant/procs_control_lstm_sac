"""YAML config loader replacing XuanCe get_arguments."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml


def load_config(config_path: str | Path, overrides: dict | None = None) -> Namespace:
    """Load a YAML config and apply CLI overrides on top."""
    with open(config_path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f'config must be a YAML mapping, got {type(data).__name__}')
    if overrides:
        data.update(overrides)
    return Namespace(**data)
