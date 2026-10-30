"""Format retrieved passages as labeled evidence for answer generation."""

from typing import Sequence, Union


def build_context(
    matches: Sequence[tuple[dict[str, Union[str, int]], float]],
) -> str:
    """Return one source-labeled block per passage, preserving retrieval order."""
    blocks = []
    for number, (chunk, _score) in enumerate(matches, start=1):
        for field in ("text", "source", "id"):
            value = chunk.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"passage {number} requires nonblank {field}")

        start = chunk.get("start")
        end = chunk.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"passage {number} requires valid character offsets")
        if end - start != len(chunk["text"]):
            raise ValueError(f"passage {number} offsets do not match its text length")

        block = (
            f"[S{number}]\n"
            f"Source: {chunk['source']}\n"
            f"Chunk ID: {chunk['id']}\n"
            f"Offsets: [{start}:{end}]\n"
            f"Text:\n{chunk['text']}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)
