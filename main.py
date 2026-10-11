"""Load and inspect a local document."""

from pathlib import Path

from rag.documents import load_document


def main() -> None:
    sample_path = Path(__file__).resolve().parent / "data" / "sample.txt"
    document = load_document(sample_path)

    print(f"Source: {document['source']}")
    print("\nText:")
    print(document["text"])


if __name__ == "__main__":
    main()
