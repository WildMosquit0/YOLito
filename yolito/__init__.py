"""
High-level entry points that make the YOLito project consumable as a Python package
without forcing callers to reach into the research scripts.
"""
from __future__ import annotations

from .config import (
    CONFIG_ENV_VAR,
    RuntimeConfig,
    TaskName,
    build_runtime_config,
    resolve_config_path,
)
from .runtime import run_task

__all__ = [
    "CONFIG_ENV_VAR",
    "RuntimeConfig",
    "TaskName",
    "build_runtime_config",
    "resolve_config_path",
    "run_task",
]

__version__ = "0.1.0"
