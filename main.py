"""Load, clean, and split a local document into inspectable chunks."""

from pathlib import Path

from rag.chunking import chunk_document
from rag.documents import load_document
from rag.preprocessing import clean_text


def main() -> None:
    sample_path = Path(__file__).resolve().parent / "data" / "sample.txt"
    document = load_document(sample_path)
    cleaned_text = clean_text(document["text"])
    cleaned_document = {"text": cleaned_text, "source": document["source"]}
    chunk_size = 120
    overlap = 20
    chunks = chunk_document(cleaned_document, chunk_size=chunk_size, overlap=overlap)

    print(f"Source: {document['source']}")
    print("\nOriginal text:")
    print(document["text"])
    print("\nCleaned text:")
    print(cleaned_text)

    print(f"\nChunks: {len(chunks)} (size={chunk_size}, overlap={overlap} characters)")
    for chunk in chunks:
        print(f"\nChunk {chunk['index']}: [{chunk['start']}:{chunk['end']}]")
        print(f"Source: {chunk['source']}")
        print(repr(chunk["text"]))


if __name__ == "__main__":
    main()
