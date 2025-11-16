from __future__ import annotations

import argparse
import cProfile
import pstats
from typing import Optional, Sequence

from .config import TaskName, build_runtime_config
from .runtime import run_task

_PROFILE_SORT_KEY = "cumtime"


def _task_type(value: str) -> TaskName:
    lowered = value.lower()
    if lowered not in ("infer", "analyze"):
        raise argparse.ArgumentTypeError(f"Unsupported task '{value}'.")
    return lowered  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosquito-supermodel",
        description="Run inference or analysis tasks with a publishable CLI surface.",
    )
    parser.add_argument(
        "--task",
        "--task",
        dest="task",
        default="infer",
        type=_task_type,
        help="Task to execute (defaults to infer). Alias: --task for backwards compatibility.",
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        help="Path to the YAML file. Defaults to configs/<task>.yaml or $MOSQUITO_SUPERMODEL_CONFIG_DIR.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        help="Enable CPU profiling (default behaviour for backwards compatibility).",
    )
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="Disable CPU profiling output.",
    )
    parser.add_argument(
        "--profile-depth",
        dest="profile_depth",
        type=int,
        default=10,
        help="Number of rows to show in the profiler table (default: 10).",
    )

    parser.set_defaults(profile=True)
    return parser


def _maybe_profile(enabled: bool, depth: int) -> cProfile.Profile | None:
    if not enabled:
        return None
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def _finish_profile(profiler: Optional[cProfile.Profile], depth: int) -> None:
    if profiler is None:
        return
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(_PROFILE_SORT_KEY)
    stats.print_stats(depth)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint used by ``python -m mosquito_supermodel`` and the console script."""
    args = parse_args(argv)
    runtime_config = build_runtime_config(args.task, args.config_path)

    profiler = _maybe_profile(args.profile, args.profile_depth)
    try:
        run_task(runtime_config)
    finally:
        _finish_profile(profiler, args.profile_depth)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
