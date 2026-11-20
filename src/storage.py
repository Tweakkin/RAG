"""Read and atomically replace JSON pipeline data."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel


def write_model(
    path: Path, model: BaseModel, *, compact: bool = False,
) -> None:
    """Replace a JSON file only after fully serializing its new contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            indent = None if compact else 2
            handle.write(model.model_dump_json(indent=indent).encode("utf-8"))
            handle.write(b"\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
