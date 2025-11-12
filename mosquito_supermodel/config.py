from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from src.utils.config_ops import load_config

TaskName = Literal["infer", "analyze"]
CONFIG_ENV_VAR = "MOSQUITO_SUPERMODEL_CONFIG_DIR"
_DEFAULT_CONFIG_DIR = "configs"


@dataclass(frozen=True)
class RuntimeConfig:
    """Small value object that pairs a task name with the YAML file it should use."""

    task: TaskName
    config_path: Path

    def load(self) -> dict:
        """Load the YAML payload as a Python dictionary."""
        return load_config(str(self.config_path))


def _project_root(anchor: Optional[Path] = None) -> Path:
    """Return the repository root by walking up until we find the configs directory."""
    if anchor is None:
        anchor = Path(__file__).resolve()

    for candidate in anchor.parents:
        if (candidate / _DEFAULT_CONFIG_DIR).exists():
            return candidate

    # Fallback to the parent of the package directory when configs/ is missing.
    return anchor.parents[1]


def _default_config_dir(anchor: Optional[Path] = None) -> Path:
    """
    Resolve the directory that contains task YAML files.

    Order of precedence:
    1. MOSQUITO_SUPERMODEL_CONFIG_DIR environment variable (allows docker/custom images)
    2. `<repo>/configs`
    """
    env_override = os.getenv(CONFIG_ENV_VAR)
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (_project_root(anchor) / _DEFAULT_CONFIG_DIR).resolve()


def resolve_config_path(
    task: TaskName,
    config_path: Optional[os.PathLike[str] | str] = None,
    anchor: Optional[Path] = None,
) -> Path:
    """
    Determine which YAML file should be used for the requested task.

    When ``config_path`` is omitted we fall back to ``<config_dir>/<task>.yaml``.
    """
    if config_path:
        candidate = Path(config_path).expanduser().resolve()
    else:
        candidate = _default_config_dir(anchor) / f"{task}.yaml"

    if not candidate.exists():
        raise FileNotFoundError(f"Configuration file not found for task '{task}': {candidate}")

    return candidate


def build_runtime_config(
    task: TaskName,
    config_path: Optional[os.PathLike[str] | str] = None,
    anchor: Optional[Path] = None,
) -> RuntimeConfig:
    """Helper that couples ``task`` with the resolved configuration path."""
    return RuntimeConfig(task=task, config_path=resolve_config_path(task, config_path, anchor))
