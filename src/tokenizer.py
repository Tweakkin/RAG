def tokenize(text: str) -> list[str]:
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
    counts: dict[str, int] = {}

    for token in tokens:
        if token in counts:
            counts[token] += 1
        else:
            counts[token] = 1

    return counts

# How many chunks contain each token
def document_frequency(chunks_tokens: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}  # Store how many chunks contain each token.

    for tokens in chunks_tokens:  # Visit each chunk's list of tokens.
        for token in set(tokens):  # Count a token only once per chunk.
            if token in counts:
                counts[token] += 1  # Another chunk contains this token.
            else:
                counts[token] = 1  # First chunk containing this token.

    return counts