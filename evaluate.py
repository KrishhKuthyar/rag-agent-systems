"""Run a small labeled retrieval evaluation with the cached local embedding model."""

import argparse
from pathlib import Path
import sys

from rag.chunking import chunk_document
from rag.documents import load_document
from rag.evaluation import evaluate_retrieval, load_evaluation_dataset, validate_evaluation_document
from rag.local_embeddings import LOCAL_MODEL, LOCAL_MODEL_REVISION, embed_texts_locally, load_local_model
from rag.preprocessing import clean_text


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate retrieval against labeled sample questions.")
    parser.add_argument("--dataset", type=Path, default=project_root / "data" / "retrieval_eval.json")
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    if sys.version_info < (3, 10):
        parser.error("local model dependencies require Python 3.10+; use .venv/bin/python")
    try:
        dataset = load_evaluation_dataset(args.dataset)
        document = load_document(args.dataset.resolve().parent / dataset["document"])
        cleaned = {"text": clean_text(document["text"]), "source": document["source"]}
        validate_evaluation_document(dataset, cleaned["text"])
        chunks = chunk_document(cleaned, chunk_size=120, overlap=20)
        cases = dataset["cases"]
        print(f"Model: {LOCAL_MODEL}; revision: {LOCAL_MODEL_REVISION}", flush=True)
        print(f"Document: {document['source']}")
        print(f"Chunks: {len(chunks)}; size=120; overlap=20 characters; top_k={args.top_k}")
        print(f"Questions: {len(cases)}; mode: cached offline evaluation", flush=True)
        model = load_local_model(project_root / ".local" / "models")
        texts = [chunk["text"] for chunk in chunks] + [case["question"] for case in cases]
        vectors = embed_texts_locally(texts, model)
        report = evaluate_retrieval(cases, chunks, vectors[:len(chunks)], vectors[len(chunks):], args.top_k)
        for row in report["rows"]:
            status = ("HIT" if row["hit"] else "MISS") if row["answerable"] else "NO SUPPORTING EVIDENCE (labeled)"
            print(f"\n{row['id']}: {row['question']}\n  {status}")
            print(f"  Relevant chunks: {row['relevant_indices']}")
            print("  Retrieved: " + ", ".join(f"chunk {index} ({score:.4f})" for index, score in row["retrieved"]))
        print(f"\nAnswerable cases: {report['answerable_count']}")
        if report["hit_rate"] is None:
            print("Hit@k and MRR@k: N/A (no answerable cases)")
        else:
            print(f"Hit@{args.top_k}: {report['hit_rate']:.4f}")
            print(f"MRR@{args.top_k}: {report['mrr']:.4f}")
        print(f"Unanswerable cases reported separately: {report['unanswerable_count']}")
        print("Retrieval baseline only: no answer generation or automatic abstention was evaluated.")
    except ImportError:
        parser.error("install local dependencies with .venv/bin/python -m pip install -r requirements-embeddings.txt")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
