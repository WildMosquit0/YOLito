from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from .config import RuntimeConfig, TaskName

if TYPE_CHECKING:
    from src.analyze.analysis_pipeline import run_analysis as _run_analysis_type
    from src.postprocess.infer_sngle_or_multi import (
        inference_single_or_multi as _inference_single_or_multi_type,
    )


class TaskExecutionError(RuntimeError):
    """Raised when an invalid or unsupported task name is requested."""


@dataclass
class TaskPayload:
    """Introspectable structure returned by :func:`RuntimeConfig.load`."""

    data: Dict
    config_path: str
    task: TaskName


def _load_payload(runtime_config: RuntimeConfig, preload: Optional[Dict] = None) -> TaskPayload:
    payload = preload if preload is not None else runtime_config.load()
    return TaskPayload(
        data=payload,
        config_path=str(runtime_config.config_path),
        task=runtime_config.task,
    )


def run_task(runtime_config: RuntimeConfig, *, preload: Optional[Dict] = None) -> None:
    """
    Execute the configured task without altering existing inference/analysis logic.

    Args:
        runtime_config: Carries the task name and resolved YAML path.
        preload: Optional config dict for callers that already loaded YAML.
    """
    payload = _load_payload(runtime_config, preload)

    if payload.task == "infer":
        from src.postprocess.infer_sngle_or_multi import inference_single_or_multi

        inference_single_or_multi(payload.data, payload.config_path)
        return

    if payload.task == "analyze":
        from src.analyze.analysis_pipeline import run_analysis

        run_analysis(payload.data, payload.config_path)
        return

    raise TaskExecutionError(f"Unsupported task '{payload.task}'.")
