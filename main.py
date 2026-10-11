"""Load a local document and inspect its text before and after cleanup."""

from pathlib import Path

from rag.documents import load_document
from rag.preprocessing import clean_text


def main() -> None:
    sample_path = Path(__file__).resolve().parent / "data" / "sample.txt"
    document = load_document(sample_path)
    cleaned_text = clean_text(document["text"])

    print(f"Source: {document['source']}")
    print("\nOriginal text:")
    print(document["text"])
    print("\nCleaned text:")
    print(cleaned_text)


if __name__ == "__main__":
    main()
