import math
import re

# These common question words keep 20% of their normal score.
LOW_WEIGHT_WORDS = {
    "what", "which", "how", "is", "are",
    "a", "an", "the", "of", "in", "to",
}

# Takes a chunk's text and returns a list of tokens
def tokenize(text: str) -> list[str]:
    """Keep complete identifiers and add their individual words."""

    # Keep letters, numbers, and underscores.
    # Preserve capitals so we can detect word boundaries.
    cleaned = ""
    for character in text:
        if character.isalnum() or character == "_":
            cleaned += character
        else:
            cleaned += " "

    # Store the resulting tokens.
    tokens: list[str] = []

    # Process each word separately.
    for word in cleaned.split():

        # Skip words containing only underscores.
        if not word.strip("_"):
            continue

        # Keep the complete identifier in lowercase.
        tokens.append(word.lower())

        # Separate an acronym from the next word.
        # "HTTPServer" -> "HTTP Server"
        separated = re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1 \2",
            word,
        )

        # Separate lowercase letters or digits from a following capital.
        # "BatchedExperts" -> "Batched Experts"
        separated = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1 \2",
            separated,
        )

        # Split underscores too, then lowercase all parts.
        parts = separated.replace("_", " ").lower().split()

        # Add the parts without duplicating an unchanged word.
        if parts != [word.lower()]:
            tokens.extend(parts)

    return tokens

def tokenize_question(text: str) -> list[str]:
    """Keep original words and selectively add identifier parts."""

    # Remove punctuation while preserving capitals and underscores.
    cleaned = ""
    for character in text:
        if character.isalnum() or character == "_":
            cleaned += character
        else:
            cleaned += " "

    tokens: list[str] = []

    for word in cleaned.split():
        # Ignore words containing only underscores.
        if not word.strip("_"):
            continue

        # Always keep the complete word.
        base_tokens = [word.lower()]

        # Keep the old underscore-splitting behavior.
        if "_" in word:
            for part in word.lower().split("_"):
                if part:
                    base_tokens.append(part)

        tokens.extend(base_tokens)

        # Get extra parts from the existing regex tokenizer.
        extra_parts = [
            part for part in tokenize(word)
            if part not in base_tokens
        ]

        # Add them only if EVERY extra part has at least 3 characters.
        if extra_parts and all(len(part) >= 3 for part in extra_parts):
            tokens.extend(extra_parts)

    return tokens

# Receive tokens, return their counts.
def term_frequency(tokens: list[str]) -> dict[str, int]:
    """Count how often each token occurs in one chunk."""
    counts: dict[str, int] = {}

    for token in tokens:
        if token in counts:
            counts[token] += 1
        else:
            counts[token] = 1

    return counts

# How many chunks contain each token
def document_frequency(chunks_tokens: list[list[str]]) -> dict[str, int]:
    """Count how many chunks contain each token, once per chunk."""
    counts: dict[str, int] = {}  # Store how many chunks contain each token.

    for tokens in chunks_tokens:  # Visit each chunk's list of tokens.
        for token in set(tokens):  # Count a token only once per chunk.
            if token in counts:
                counts[token] += 1  # Another chunk contains this token.
            else:
                counts[token] = 1  # First chunk containing this token.

    return counts


# Returns each token and its weight
def inverse_document_frequency(
    df: dict[str, int],
    total_chunks: int
) -> dict[str, float]:
    """Calculate smoothed IDF weights from chunk document frequencies."""
    idf: dict[str, float] = {}

    for word, count in df.items():
        idf[word] = math.log(
            (total_chunks + 1) / (count + 1)
        ) + 1
    return idf


def prepare_chunks(
    chunks_dict: list[dict[str, str | int]],
) -> tuple[list[dict[str, int]], dict[str, float]]:
    """Prepare corpus statistics once for reuse across search questions.

    Args:
        chunks_dict: Ordered chunks containing a string in the 'text' field.

    Returns:
        A tuple containing token counts for each chunk, in input order,
        and the collection-wide smoothed IDF weights.

    Raises:
        ValueError: If a chunk's text is not a string.
        KeyError: If a chunk has no 'text' field.
    """
    chunks_tokens: list[list[str]] = []
    for chunk in chunks_dict:
        chunk_text = chunk["text"]
        if not isinstance(chunk_text, str):
            raise ValueError("Each chunk's text must be a string.")
        chunks_tokens.append(tokenize(chunk_text))

    # Count each word's occurrences separately inside each chunk.
    chunks_term_frequencies: list[dict[str, int]] = []
    for chunk_tokens in chunks_tokens:
        chunks_term_frequencies.append(term_frequency(chunk_tokens))

    # Calculate collection-wide DF, then smoothed IDF weights.
    document_frequencies = document_frequency(chunks_tokens)
    total_chunks = len(chunks_dict)
    inverse_document_frequencies = inverse_document_frequency(
        document_frequencies, total_chunks
    )
    return chunks_term_frequencies, inverse_document_frequencies


def score_chunks(
    question: str,
    chunks_term_frequencies: list[dict[str, int]],
    inverse_document_frequencies: dict[str, float],
) -> list[float]:
    """Calculate a simple TF-IDF search score for each chunk.

    Args:
        question: The search question, before tokenization.
        chunks_term_frequencies: Per-chunk token counts from prepare_chunks.
        inverse_document_frequencies: IDF weights from prepare_chunks.

    Returns:
        One score per chunk, in the same order as the input chunks.
        An empty question gives zero scores; no chunks gives an empty list.
        Scores sum logarithmic chunk TF times smoothed IDF for unique query tokens.
        Common question words contribute 20% of their normal value.
    """
    # Expand question identifiers selectively to avoid noisy short parts.
    question_tokens = set(tokenize_question(question))

    # Score every chunk using only the words from the question.
    chunk_scores: list[float] = []
    for chunk_term_frequencies in chunks_term_frequencies:
        chunk_score = 0.0
        for question_token in question_tokens:
            # A missing word contributes zero to this chunk's score.
            raw_tf = chunk_term_frequencies.get(question_token, 0)
            word_idf = inverse_document_frequencies.get(question_token, 0.0)
            tf = 1 + math.log(raw_tf) if raw_tf > 0 else 0.0
            contribution = tf * word_idf

            # Reduce the influence of common question words.
            if question_token in LOW_WEIGHT_WORDS:
                contribution *= 0.2

            chunk_score += contribution
        chunk_scores.append(chunk_score)

    return chunk_scores

def get_top_chunks(
    chunks: list[dict[str, str | int]],
    scores: list[float],
    k: int = 5,
) -> list[dict[str, str | int]]:
    """Return up to k matching chunks, ranked by descending score."""
    if len(chunks) != len(scores):
        raise ValueError("Each chunk must have exactly one score.")

    if k <= 0:
        return []

    # Keep each positive score connected to its chunk's index.
    scored_indices: list[tuple[float, int]] = []

    for chunk_index in range(len(chunks)):
        score = scores[chunk_index]

        if score > 0:
            scored_indices.append((score, chunk_index))

    # Sort by score, highest first.
    scored_indices.sort(key=lambda item: item[0], reverse=True)

    # Use the saved indices to retrieve the best chunks.
    top_chunks: list[dict[str, str | int]] = []

    for score, chunk_index in scored_indices[:k]:
        top_chunks.append(chunks[chunk_index])

    return top_chunks
