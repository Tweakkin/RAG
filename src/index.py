"""Build and load the persisted lexical index."""

import sys
from collections.abc import Callable, Iterable
from hashlib import sha256
from pathlib import Path

from .chunking import chunk_file, validate_chunk_size
from .ingestion import corpus_files, read_source, source_name
from .models import IndexData, IndexedFile
from .storage import write_model
from .tokenizer import term_counts

INDEX_NAME = "rag_index.json"
INDEX_VERSION = 1


def load_index(path: Path) -> IndexData:
    """Load a validated index with a clear error for missing or old caches."""
    if not path.is_file():
        raise FileNotFoundError(f"Index not found: {path}. Run index first.")
    try:
        index = IndexData.model_validate_json(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"Invalid index: {path}. Run index again.") from error
    if index.version != INDEX_VERSION:
        raise ValueError(f"Outdated index: {path}. Run index again.")
    for name, entry in index.files.items():
        if len(entry.chunks) != len(entry.term_counts):
            raise ValueError(f"Incomplete index: {path}. Run index again.")
        for chunk in entry.chunks:
            length = chunk.last_character_index - chunk.first_character_index
            if (chunk.file_path != name or length != len(chunk.text)
                    or length > index.max_chunk_size):
                raise ValueError(f"Invalid chunk in {path}. Run index again.")
    return index


def build_index(
    corpus_path: Path,
    index_path: Path,
    project_root: Path,
    size: int,
    progress: Callable[[list[Path]], Iterable[Path]],
) -> tuple[int, int, int]:
    """Update changed files and drop entries for files that were removed."""
    validate_chunk_size(size)
    files = corpus_files(corpus_path)
    corpus = corpus_path.resolve().as_posix()
    previous: dict[str, IndexedFile] = {}
    try:
        cache = load_index(index_path)
    except (FileNotFoundError, ValueError):
        pass
    else:
        if cache.corpus == corpus and cache.max_chunk_size == size:
            previous = cache.files

    entries: dict[str, IndexedFile] = {}
    updated = 0
    for file in progress(files):
        name = source_name(file, project_root)
        try:
            contents = read_source(file)
        except (OSError, UnicodeError) as error:
            print(f"Skipping {name}: {error}", file=sys.stderr)
            continue
        digest = sha256(contents.encode("utf-8")).hexdigest()
        cached = previous.get(name)
        if cached is not None and cached.digest == digest:
            entries[name] = cached
            continue
        chunks = chunk_file(name, contents, size)
        entries[name] = IndexedFile(
            digest=digest,
            chunks=chunks,
            term_counts=[term_counts(chunk.text) for chunk in chunks],
        )
        updated += 1

    if updated or not index_path.is_file() or entries != previous or not files:
        write_model(index_path, IndexData(
            version=INDEX_VERSION,
            corpus=corpus,
            max_chunk_size=size,
            files=entries,
        ), compact=True)
    chunk_count = sum(len(item.chunks) for item in entries.values())
    return len(entries), chunk_count, updated
