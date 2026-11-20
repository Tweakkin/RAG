"""Search chunks with an inverted, identifier-aware TF-IDF index."""

from collections import defaultdict
from heapq import nlargest
from math import log
from pathlib import PurePosixPath

from .models import Chunk, IndexData, MinimalSource
from .tokenizer import STOP_WORDS, query_tokens, tokenize


def validate_k(k: int) -> None:
    """Require a nonnegative integer result limit."""
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer.")


def location(chunk: Chunk) -> MinimalSource:
    """Return a chunk location without serializing its text."""
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.first_character_index,
        last_character_index=chunk.last_character_index,
    )


class Retriever:
    """Prepare reusable postings for fast single-query and batch retrieval."""

    def __init__(self, index: IndexData) -> None:
        """Build sparse postings and precompute document-side weights."""
        self.chunks: list[Chunk] = []
        self.path_terms: dict[str, set[str]] = {}
        self.expansion_weights: list[float] = []
        counts: list[dict[str, int]] = []
        for path, entry in index.files.items():
            self.path_terms[path] = set(tokenize(PurePosixPath(path).stem))
            expansion = 0.4 if path.endswith(".py") else 0.22
            self.expansion_weights.extend([expansion] * len(entry.chunks))
            self.chunks.extend(entry.chunks)
            counts.extend(entry.term_counts)

        lengths = [sum(chunk.values()) for chunk in counts]
        average = sum(lengths) / max(len(lengths), 1) or 1
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for number, (terms, length) in enumerate(zip(counts, lengths)):
            norm = (0.65 + length / average) ** 0.13
            for token, count in terms.items():
                if count > 0:
                    weight = (1 + log(count)) / norm
                    self.postings[token].append((number, weight))

        self.idf = {
            token: log((len(self.chunks) + 1) / (len(entries) + 1)) + 1
            for token, entries in self.postings.items()
        }

    def search(self, question: str, k: int) -> list[Chunk]:
        """Rank matching chunks with stable ties and a filename signal."""
        validate_k(k)
        if not isinstance(question, str):
            raise ValueError("query must be text.")
        literal, expanded, symbols = query_tokens(question)
        terms = literal | expanded
        if not k or not ((terms - STOP_WORDS) & self.idf.keys()):
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in sorted(terms, key=lambda word: (word in STOP_WORDS, word)):
            if term not in self.idf:
                continue
            weight = self.idf[term]
            if term in STOP_WORDS:
                weight *= 0.13
            elif term in symbols:
                weight *= 1.45
            for number, tf in self.postings[term]:
                if term in STOP_WORDS and number not in scores:
                    continue
                multiplier = (self.expansion_weights[number]
                              if term in expanded else 1.0)
                scores[number] += weight * tf * multiplier

        candidates = [number for number in scores if scores[number] > 0]
        path_boost: dict[str, float] = {}
        identifying_terms = literal - STOP_WORDS
        for number in candidates:
            path = self.chunks[number].file_path
            if path not in path_boost:
                matching = self.path_terms[path] & identifying_terms
                path_boost[path] = 0.8 * sum(
                    self.idf.get(term, 0.0) for term in sorted(matching)
                )
            scores[number] += path_boost[path]
        best = nlargest(k, candidates, key=lambda n: (scores[n], -n))
        return [self.chunks[number] for number in best]
