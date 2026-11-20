"""Read supported source files without changing their character offsets."""

import os
from pathlib import Path


def read_source(path: Path) -> str:
    """Decode UTF-8 source without translating line endings."""
    return path.read_bytes().decode("utf-8")


def corpus_files(root: Path) -> list[Path]:
    """Find supported non-symlinked files in a deterministic order."""
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")

    def raise_walk_error(error: OSError) -> None:
        """Report a directory that could not be read."""
        raise error

    files: list[Path] = []
    for folder, subfolders, names in os.walk(root, onerror=raise_walk_error):
        subfolders[:] = [name for name in subfolders if name != ".git"]
        for name in names:
            path = Path(folder) / name
            if path.suffix in {".py", ".md", ".txt"} and not path.is_symlink():
                files.append(path)
    return sorted(files)


def source_name(path: Path, project_root: Path) -> str:
    """Use project-relative, forward-slash paths where possible."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
