"""Compare vector directions using cosine similarity."""

import math
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return a score in [-1, 1] for equal-length, finite, nonzero vectors."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("vectors must not be empty")
    if len(a) != len(b):
        raise ValueError("vectors must have equal dimensions")
    for vector in (a, b):
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("vector coordinates must be finite")

    scale_a = max(abs(value) for value in a)
    scale_b = max(abs(value) for value in b)
    if scale_a == 0 or scale_b == 0:
        raise ValueError("zero vectors have no direction")

    # Positive rescaling preserves direction and avoids overflow or underflow.
    scaled_a = [value / scale_a for value in a]
    scaled_b = [value / scale_b for value in b]
    dot_product = math.fsum(x * y for x, y in zip(scaled_a, scaled_b))
    length_a = math.sqrt(math.fsum(value * value for value in scaled_a))
    length_b = math.sqrt(math.fsum(value * value for value in scaled_b))
    score = dot_product / (length_a * length_b)

    # Floating-point rounding can otherwise put a score just outside [-1, 1].
    return max(-1.0, min(1.0, score))
