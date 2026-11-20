"""Create Python and prose chunks with exact source offsets."""

import ast
import re
from collections.abc import Iterator

from .models import Chunk, MAX_CHUNK_SIZE


def validate_chunk_size(size: int) -> None:
    """Require a chunk size that respects the grading limit."""
    if type(size) is not int or not 1 <= size <= MAX_CHUNK_SIZE:
        raise ValueError("max_chunk_size must be an integer from 1 to 2000.")


def paragraph_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield paragraph spans while preserving original line endings."""
    start = 0
    for match in re.finditer(r"(?:\r?\n[ \t]*){2,}", text):
        yield start, match.start()
        start = match.end()
    yield start, len(text)


def python_spans(text: str) -> Iterator[tuple[int, int]]:
    """Find top-level Python statements, including their decorators."""
    try:
        module = ast.parse(text.removeprefix("\ufeff"))
    except (SyntaxError, ValueError, RecursionError):
        yield from paragraph_spans(text)
        return

    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer(r"\r\n?|\n", text))
    line_starts.append(len(text))
    end = 0
    for node in module.body:
        first_line = node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.decorator_list:
            first_line = node.decorator_list[0].lineno
        start = line_starts[first_line - 1]
        node_end = line_starts[node.end_lineno or node.lineno]
        if start < end:
            continue
        # Comments between statements can contain useful explanations.
        if text[end:start].strip():
            yield end, start
        yield start, node_end
        end = node_end
    if text[end:].strip():
        yield end, len(text)


def chunk_file(path: str, text: str, size: int) -> list[Chunk]:
    """Group prose paragraphs and split large blocks to the requested size."""
    validate_chunk_size(size)
    is_python = path.endswith(".py")
    spans = python_spans(text) if is_python else paragraph_spans(text)
    bounds: list[tuple[int, int]] = []
    for start, end in spans:
        for left in range(start, end, size):
            right = min(left + size, end)
            if not text[left:right].strip():
                continue
            if not is_python and bounds and right - bounds[-1][0] <= size:
                bounds[-1] = bounds[-1][0], right
            else:
                bounds.append((left, right))
    return [
        Chunk(
            file_path=path,
            first_character_index=start,
            last_character_index=end,
            text=text[start:end],
        )
        for start, end in bounds
    ]
