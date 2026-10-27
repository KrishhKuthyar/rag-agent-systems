"""Request consistent text embeddings and validate their input mapping."""

import math
from typing import Any, Sequence


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def embed_texts(texts: Sequence[str], client: Any) -> list[list[float]]:
    """Return one valid vector per text, preserving input order."""
    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must be a sequence of strings, not one string")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("each input text must be a nonblank string")
    if len(texts) == 0:
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=list(texts),
        dimensions=EMBEDDING_DIMENSIONS,
        encoding_format="float",
    )
    if response.model != EMBEDDING_MODEL:
        raise ValueError("response model does not match the requested model")
    if len(response.data) != len(texts):
        raise ValueError("response must contain one embedding per input text")

    indexes = [item.index for item in response.data]
    if any(type(index) is not int for index in indexes):
        raise ValueError("response indexes must be integers")
    if sorted(indexes) != list(range(len(texts))):
        raise ValueError("response indexes must cover every input exactly once")

    vectors = []
    for item in sorted(response.data, key=lambda result: result.index):
        vector = item.embedding
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"every embedding must have {EMBEDDING_DIMENSIONS} dimensions")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in vector
        ):
            raise ValueError("embedding coordinates must be finite numbers")
        if not any(vector):
            raise ValueError("zero embeddings have no usable direction")
        vectors.append([float(value) for value in vector])

    return vectors
