"""Generate text embeddings locally with a pinned MiniLM model."""

import math
from pathlib import Path
from typing import Any, Sequence


LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LOCAL_DIMENSIONS = 384


def load_local_model(cache_dir: Path, allow_download: bool = False) -> Any:
    """Load the pinned model on CPU; use cached files unless download is requested."""
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        LOCAL_MODEL,
        revision=LOCAL_MODEL_REVISION,
        cache_folder=str(cache_dir),
        device="cpu",
        local_files_only=not allow_download,
        token=False,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    if model.get_embedding_dimension() != LOCAL_DIMENSIONS:
        raise ValueError("loaded model does not have the expected 384 dimensions")
    return model


def embed_texts_locally(texts: Sequence[str], model: Any) -> list[list[float]]:
    """Encode every input without truncation and return validated vectors in order."""
    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must be a sequence of strings, not one string")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("each input text must be a nonblank string")
    if len(texts) == 0:
        return []

    tokens = model.tokenizer(list(texts), truncation=False, add_special_tokens=True)
    input_ids = tokens["input_ids"]
    if len(input_ids) != len(texts):
        raise ValueError("tokenizer must return one sequence per input text")
    for index, token_ids in enumerate(input_ids):
        if len(token_ids) > model.max_seq_length:
            raise ValueError(
                f"input {index} has {len(token_ids)} tokens; "
                f"model limit is {model.max_seq_length}. Shorten the input."
            )

    encoded = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = encoded.tolist()
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("model must return one embedding per input text")

    checked_vectors = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != LOCAL_DIMENSIONS:
            raise ValueError("every local embedding must have 384 dimensions")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in vector
        ):
            raise ValueError("embedding coordinates must be finite numbers")
        if not any(vector):
            raise ValueError("zero embeddings have no usable direction")
        checked_vectors.append([float(value) for value in vector])

    return checked_vectors
