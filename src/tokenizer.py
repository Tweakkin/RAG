"""Tokenize natural language and source-code identifiers."""

import re
from collections import Counter

WORD = re.compile(r"[^\W_][\w]*|_+[^\W_][\w]*")
ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
CAMEL_CASE = re.compile(r"([a-z0-9])([A-Z])")
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "by", "can", "does", "for", "from", "how",
    "in", "is", "it", "of", "s", "that", "the", "to", "used", "using",
    "vllm", "what", "when", "which", "with",
})


def split_identifier(word: str) -> list[str]:
    """Split snake_case, camelCase, and acronyms into component words."""
    separated = ACRONYM.sub(r"\1 \2", word)
    separated = CAMEL_CASE.sub(r"\1 \2", separated)
    return separated.replace("_", " ").lower().split()


def tokenize(text: str) -> list[str]:
    """Retain full identifiers and index their constituent words."""
    tokens: list[str] = []
    for word in WORD.findall(text):
        lower = word.lower()
        tokens.append(lower)
        parts = split_identifier(word)
        if parts != [lower]:
            tokens.extend(part for part in parts if len(part) > 1)
    return tokens


def term_counts(text: str) -> dict[str, int]:
    """Count occurrences of each token in a snippet."""
    return dict(Counter(tokenize(text)))


def query_tokens(text: str) -> tuple[set[str], set[str], set[str]]:
    """Separate literal terms, identifier parts, and distinctive symbols."""
    words = WORD.findall(text)
    literal = {word.lower() for word in words}
    parts = {part for word in words for part in split_identifier(word)
             if len(part) > 1}
    symbols = {
        word.lower() for word in words
        if "_" in word or CAMEL_CASE.search(word)
        or (len(word) > 3 and word.isupper())
    }
    return literal, parts - literal, symbols - STOP_WORDS
