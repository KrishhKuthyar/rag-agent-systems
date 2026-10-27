"""Inspect document ingestion and demonstrate cosine similarity with toy vectors."""

from pathlib import Path

from rag.chunking import chunk_document
from rag.documents import load_document
from rag.preprocessing import clean_text
from rag.similarity import cosine_similarity


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
        print(f"ID: {chunk['id']}")
        print(f"Source: {chunk['source']}")
        print(repr(chunk["text"]))

    query_vector = [1.0, 0.0]
    candidate_vectors = {
        "same direction": [2.0, 0.0],
        "diagonal": [1.0, 1.0],
        "perpendicular": [0.0, 1.0],
        "opposite direction": [-1.0, 0.0],
    }
    print("\nCosine similarity demo (hand-written vectors):")
    print(f"Query: {query_vector}")
    for label, candidate_vector in candidate_vectors.items():
        score = cosine_similarity(query_vector, candidate_vector)
        print(f"{label}: {candidate_vector} -> {score:.4f}")


if __name__ == "__main__":
    main()
