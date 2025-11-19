"""Allow ``python -m yolito`` to behave like the CLI entry point."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
