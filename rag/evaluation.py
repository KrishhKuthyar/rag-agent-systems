"""Evaluate retrieval against explicit evidence labels, without judging answers."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from rag.retrieval import filter_by_similarity, rank_chunks, validate_min_similarity


def load_evaluation_dataset(path: Path) -> dict[str, Any]:
    """Read and validate the local question/evidence dataset."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("version") != 1:
        raise ValueError("evaluation dataset must have version 1")
    for field in ("document", "cleaned_sha256"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise ValueError(f"evaluation dataset requires {field}")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation dataset requires a nonempty cases list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every evaluation case must be a record")
        for field in ("id", "question"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"each evaluation case requires nonblank {field}")
        if case["id"] in seen:
            raise ValueError("evaluation case IDs must be unique")
        seen.add(case["id"])
        evidence = case.get("evidence")
        if not isinstance(evidence, list) or any(
            not isinstance(quote, str) or not quote.strip() for quote in evidence
        ):
            raise ValueError("evidence must be a list of nonblank quotes, or []")
    return dataset


def validate_evaluation_document(dataset: dict[str, Any], cleaned_text: str) -> None:
    """Reject stale document labels before running the model."""
    fingerprint = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    if fingerprint != dataset["cleaned_sha256"]:
        raise ValueError("cleaned document changed; review evidence labels and update its fingerprint")
    for case in dataset["cases"]:
        for quote in case["evidence"]:
            if quote not in cleaned_text:
                raise ValueError(f"case {case['id']} has evidence missing from the document")


def evaluate_retrieval(
    cases: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Union[str, int]]],
    chunk_vectors: Sequence[Sequence[float]],
    query_vectors: Sequence[Sequence[float]],
    top_k: int = 1,
    min_similarity: Optional[float] = None,
) -> dict[str, Any]:
    """Score answerable cases; report unanswerable cases without counting a pass."""
    if not cases:
        raise ValueError("at least one evaluation case is required")
    if len(cases) != len(query_vectors):
        raise ValueError("each evaluation case must have exactly one query embedding")
    if min_similarity is not None:
        validate_min_similarity(min_similarity)
    rows = []
    for case, query_vector in zip(cases, query_vectors):
        ranked = rank_chunks(chunks, chunk_vectors, query_vector, top_k=top_k)
        matches = filter_by_similarity(ranked, min_similarity) if min_similarity is not None else ranked
        relevant = {
            chunk["index"] for chunk in chunks
            if any(quote in chunk["text"] for quote in case["evidence"])
        }
        first_rank = None
        for rank, (chunk, _score) in enumerate(matches, start=1):
            if chunk["index"] in relevant:
                first_rank = rank
                break
        answerable = bool(case["evidence"])
        rows.append({
            "id": case["id"], "question": case["question"],
            "answerable": answerable, "relevant_indices": sorted(relevant),
            "retrieved": [(chunk["index"], score) for chunk, score in ranked],
            "selected": [(chunk["index"], score) for chunk, score in matches],
            "abstained": not matches,
            "hit": first_rank is not None if answerable else None,
            "reciprocal_rank": (1.0 / first_rank if first_rank else 0.0) if answerable else None,
        })
    scored = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    return {
        "rows": rows, "answerable_count": len(scored),
        "unanswerable_count": len(rows) - len(scored),
        "hit_rate": sum(row["hit"] for row in scored) / len(scored) if scored else None,
        "mrr": sum(row["reciprocal_rank"] for row in scored) / len(scored) if scored else None,
        "unsupported_accepted": sum(not row["abstained"] for row in unanswerable),
        "unsupported_blocked": sum(row["abstained"] for row in unanswerable),
        "supported_blocked": sum(row["abstained"] for row in scored),
    }
