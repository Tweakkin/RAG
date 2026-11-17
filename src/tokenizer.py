import math

# Takes a chunk's text and returns a list of tokens
def tokenize(text: str) -> list[str]:
    """Lowercase text and retain identifiers plus their underscore parts."""
    cleaned = ""

    for character in text.lower():
        # Keep letters, Numbers, underscores
        if character.isalnum() or character == "_":
            cleaned += character
        else:
            cleaned += " "

    words = cleaned.split()
    tokens: list[str] = []

    for word in words:
        if not word.strip("_"): # Ignore empty strings
            continue

        tokens.append(word)

        if "_" in word:
            for part in word.split("_"):
                if part: # Keep non-empty
                    tokens.append(part)

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
    """
    # Use the same tokenizer for the question.
    question_tokens = set(tokenize(question))

    # Score every chunk using only the words from the question.
    chunk_scores: list[float] = []
    for chunk_term_frequencies in chunks_term_frequencies:
        chunk_score = 0.0
        for question_token in question_tokens:
            # A missing word contributes zero to this chunk's score.
            word_count = chunk_term_frequencies.get(question_token, 0)
            word_idf = inverse_document_frequencies.get(question_token, 0.0)
            tf = 1 + math.log(word_count) if word_count > 0 else 0.0
            chunk_score += tf * word_idf
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
