"""Split document text into overlapping passages with source metadata."""

import hashlib
import json


def create_chunk_id(source: str, start: int, end: int, text: str) -> str:
    """Fingerprint a source passage using a fixed, versioned JSON record."""
    record = {
        "version": 1,
        "source": source,
        "start": start,
        "end": end,
        "text": text,
    }
    payload = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_document(
    document: dict[str, str],
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[dict[str, str | int]]:
    """Return character-based chunks with positions in the supplied text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = document["text"]
    source = document["source"]
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            {
                "id": create_chunk_id(source, start, end, chunk_text),
                "text": chunk_text,
                "source": source,
                "index": len(chunks),
                "start": start,
                "end": end,
            }
        )

        # Stop at the end instead of emitting another overlapping tail.
        if end == len(text):
            break
        start = end - overlap

    return chunks
