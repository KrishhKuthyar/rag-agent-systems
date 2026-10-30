"""Answer a question from local sample passages using two free local models."""

import argparse
from pathlib import Path
import sys
import time

from rag.chunking import chunk_document
from rag.context import build_context
from rag.documents import load_document
from rag.generation import (
    ANSWER_MODEL, ANSWER_MODEL_REVISION, check_answer_citations,
    generate_answer, load_answer_model,
)
from rag.local_embeddings import embed_texts_locally, load_local_model
from rag.preprocessing import clean_text
from rag.prompting import INSUFFICIENT_EVIDENCE, build_answer_messages
from rag.retrieval import rank_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer from sample passages with free local models.")
    parser.add_argument("--query", default="Why do chunks overlap?")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--download-models", action="store_true", help="allow initial public model downloads")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must contain text")
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256")
    if sys.version_info < (3, 10):
        parser.error("local models require Python 3.10+; use .venv/bin/python")

    project_root = Path(__file__).resolve().parent
    cache = project_root / ".local" / "models"
    document = load_document(project_root / "data" / "sample.txt")
    cleaned = {"text": clean_text(document["text"]), "source": document["source"]}
    chunks = chunk_document(cleaned, chunk_size=120, overlap=20)
    print(f"Question: {args.query}", flush=True)
    print(f"Source: {document['source']}")
    print(f"Chunks: {len(chunks)}; size=120; overlap=20 characters; top_k={args.top_k}")
    print("Mode: public downloads allowed" if args.download_models else "Mode: cached offline inference", flush=True)
    try:
        if not chunks:
            print(f"Answer: {INSUFFICIENT_EVIDENCE}")
            print("No passages available; no models were loaded.")
            return

        embedding_model = load_local_model(cache, allow_download=args.download_models)
        texts = [chunk["text"] for chunk in chunks] + [args.query]
        vectors = embed_texts_locally(texts, embedding_model)
        del embedding_model
        matches = rank_chunks(chunks, vectors[:-1], vectors[-1], top_k=args.top_k)
        context = build_context(matches)
        messages = build_answer_messages(args.query, context)
        print("\nRetrieved context:")
        print(context)
        if args.show_prompt:
            for message in messages:
                print(f"\n{message['role'].upper()}:\n{message['content']}")

        print(f"\nAnswer model: {ANSWER_MODEL}")
        print(f"Revision: {ANSWER_MODEL_REVISION}; CPU; max_new_tokens={args.max_new_tokens}", flush=True)
        tokenizer, model = load_answer_model(cache, allow_download=args.download_models)
        started = time.perf_counter()
        answer = generate_answer(messages, tokenizer, model, args.max_new_tokens)
        print(f"\nGenerated answer:\n{answer}", flush=True)
        check_answer_citations(answer, len(matches))
        print("Citation labels checked against retrieved sources; factual support needs review.")
        print(f"Generation time: {time.perf_counter() - started:.1f} seconds")
    except ImportError:
        parser.error("install local dependencies: .venv/bin/python -m pip install -r requirements-embeddings.txt")
    except OSError:
        parser.error("could not load model files; for initial setup, rerun with --download-models")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
