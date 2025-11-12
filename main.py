from mosquito_supermodel.cli import main as cli_entrypoint
from mosquito_supermodel.config import build_runtime_config
from mosquito_supermodel.runtime import run_task


def main(task: str) -> None:
    """
    Backwards-compatible helper so existing notebooks/scripts calling ``main("infer")``
    continue to behave exactly the same after the refactor.
    """
    runtime_config = build_runtime_config(task)
    run_task(runtime_config)


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
