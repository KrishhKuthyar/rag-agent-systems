"""Load document text while retaining its source."""

from pathlib import Path


def load_document(path: Path) -> dict[str, str]:
    """Read UTF-8 text and keep its absolute source path."""
    text = path.read_text(encoding="utf-8")
    return {"text": text, "source": str(path.resolve())}
