"""Embed sample passages and a question locally without paid API calls."""

import argparse
from pathlib import Path
import sys

from rag.chunking import chunk_document
from rag.documents import load_document
from rag.local_embeddings import (
    LOCAL_DIMENSIONS,
    LOCAL_MODEL,
    LOCAL_MODEL_REVISION,
    embed_texts_locally,
    load_local_model,
)
from rag.preprocessing import clean_text
from rag.similarity import cosine_similarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed sample passages and a question with a free local MiniLM model."
    )
    parser.add_argument("--query", default="Why do chunks overlap?")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="allow the initial public model download; otherwise use cached files offline",
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must contain text")
    if sys.version_info < (3, 10):
        parser.error("local model dependencies require Python 3.10+; use .venv/bin/python")

    project_root = Path(__file__).resolve().parent
    document = load_document(project_root / "data" / "sample.txt")
    cleaned_document = {
        "text": clean_text(document["text"]),
        "source": document["source"],
    }
    chunks = chunk_document(cleaned_document, chunk_size=120, overlap=20)
    texts = [chunk["text"] for chunk in chunks] + [args.query]

    print(f"Local model: {LOCAL_MODEL}", flush=True)
    print(f"Revision: {LOCAL_MODEL_REVISION}")
    print(f"Dimensions: {LOCAL_DIMENSIONS}; device: CPU")
    print("Mode: public download allowed" if args.download_model else "Mode: cached offline inference")
    print(f"Source: {document['source']}")
    for index, text in enumerate(texts):
        label = "query" if index == len(chunks) else f"chunk {index}"
        print(f"Input {index} ({label}): {text!r}")

    try:
        model = load_local_model(
            project_root / ".local" / "models",
            allow_download=args.download_model,
        )
        vectors = embed_texts_locally(texts, model)
    except ImportError:
        parser.error("install dependencies with: .venv/bin/python -m pip install -r requirements-embeddings.txt")
    except OSError:
        parser.error("could not load model files; for the initial download, rerun with --download-model")
    except ValueError as error:
        parser.error(str(error))

    query_vector = vectors[-1]
    print(f"\nAccepted local embeddings: {len(vectors)}")
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
