"""Normalize whitespace in prose while preserving paragraph boundaries."""

import re


def clean_text(text: str) -> str:
    """Normalize line endings, spaces, tabs, and repeated blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean each line separately so spaces and tabs do not consume newlines.
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip(" ")
        lines.append(cleaned_line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")
