"""Optional paid OpenAI example; the default embedding demo uses a local model."""

import argparse
import os
from pathlib import Path
import sys

from rag.chunking import chunk_document
from rag.documents import load_document
from rag.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_texts
from rag.preprocessing import clean_text
from rag.similarity import cosine_similarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed sample passages and a question using the OpenAI API."
    )
    parser.add_argument("--query", default="Why do chunks overlap?")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must contain text")
    if sys.version_info < (3, 10):
        parser.error("the embedding SDK requires Python 3.10+; use .venv/bin/python")

    try:
        from dotenv import load_dotenv
        from openai import APIError, OpenAI
    except ImportError:
        parser.error("install dependencies with: .venv/bin/python -m pip install -r requirements-openai.txt")

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        parser.error("set OPENAI_API_KEY in your environment or this project's ignored .env file")

    document = load_document(project_root / "data" / "sample.txt")
    cleaned_document = {
        "text": clean_text(document["text"]),
        "source": document["source"],
    }
    chunks = chunk_document(cleaned_document, chunk_size=120, overlap=20)
    texts = [chunk["text"] for chunk in chunks] + [args.query]

    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Requested dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Source: {document['source']}")
    for index, text in enumerate(texts):
        label = "query" if index == len(chunks) else f"chunk {index}"
        print(f"Input {index} ({label}): {text!r}")

    try:
        with OpenAI(api_key=api_key, timeout=30.0, max_retries=0) as client:
            vectors = embed_texts(texts, client)
    except APIError as error:
        if getattr(error, "code", None) == "credit_balance_exhausted":
            raise SystemExit(
                "OpenAI API credit balance is exhausted (credit_balance_exhausted). "
                "Add credits in your OpenAI API billing settings before retrying."
            ) from None
        raise SystemExit(
            f"OpenAI request failed ({type(error).__name__}). "
            "Check your API key, API access/quota, and network connection."
        ) from None
    except ValueError as error:
        raise SystemExit(f"Embedding response rejected: {error}") from None

    query_vector = vectors[-1]
    print(f"\nAccepted embeddings: {len(vectors)}")
    print(f"Query: {args.query!r}")
    print(f"Query dimensions: {len(query_vector)}")
    print(f"Query vector preview (first 5): {query_vector[:5]}")
    for chunk, vector in zip(chunks, vectors[:-1]):
        score = cosine_similarity(query_vector, vector)
        print(f"\nChunk {chunk['index']}: [{chunk['start']}:{chunk['end']}]")
        print(f"ID: {chunk['id']}")
        print(f"Text: {chunk['text']!r}")
        print(f"Dimensions: {len(vector)}")
        print(f"Vector preview (first 5): {vector[:5]}")
        print(f"Cosine similarity to query: {score:.4f}")


if __name__ == "__main__":
    main()
