"""Select the highest-scoring passages using cosine similarity."""

import math
from typing import Sequence, Union

from rag.similarity import cosine_similarity


DEFAULT_MIN_SIMILARITY = 0.25


def validate_min_similarity(value: float) -> None:
    """Require a finite cosine cutoff in the mathematical score range."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not -1 <= value <= 1
    ):
        raise ValueError("min_similarity must be a finite number between -1 and 1")


def filter_by_similarity(
    matches: Sequence[tuple[dict[str, Union[str, int]], float]],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[tuple[dict[str, Union[str, int]], float]]:
    """Keep qualifying passage copies in their existing order; do not infer factual support."""
    validate_min_similarity(min_similarity)
    selected = []
    for chunk, score in matches:
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not -1 <= score <= 1
        ):
            raise ValueError("passage similarity must be a finite number between -1 and 1")
        if score >= min_similarity:
            selected.append((chunk.copy(), score))
    return selected


def rank_chunks(
    chunks: Sequence[dict[str, Union[str, int]]],
    chunk_vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    top_k: int = 2,
) -> list[tuple[dict[str, Union[str, int]], float]]:
    """Return up to top_k (chunk copy, score) pairs, highest score first."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if len(chunks) != len(chunk_vectors):
        raise ValueError("each chunk must have exactly one embedding")

    # Validate the query even when the document contains no passages.
    cosine_similarity(query_vector, query_vector)
    scored_chunks = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, vector)
        scored_chunks.append((chunk.copy(), score))

    # Python's stable sort keeps input order when scores tie.
    ranked_chunks = sorted(scored_chunks, key=lambda result: result[1], reverse=True)
    return ranked_chunks[:top_k]
