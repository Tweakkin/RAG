"""Run the RAG command-line interface."""

import sys
from typing import NoReturn


def fail(message: str, status: int = 1) -> NoReturn:
    """Report a command failure on stderr without a traceback."""
    print(message, file=sys.stderr)
    raise SystemExit(status)


def main() -> None:
    """Dispatch a Fire command and render operational errors concisely."""
    try:
        import fire

        from .cli import RagCLI

        fire.Fire(RagCLI, name="rag")
    except KeyboardInterrupt:
        fail("Interrupted.", 130)
    except ImportError as error:
        fail(f"RAG error: {error}. Install dependencies with uv sync.")
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        fail(f"RAG error: {error}")


if __name__ == "__main__":
    main()
