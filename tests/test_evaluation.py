"""Check retrieval metrics, negative labels, and evaluation input integrity."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import evaluate
from rag.evaluation import evaluate_retrieval, load_evaluation_dataset, validate_evaluation_document


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "one", "question": "Question?", "evidence": ["evidence"]}]
        self.chunks = [{"index": 0, "text": "unrelated"}, {"index": 1, "text": "evidence"}]
        self.vectors = [[1.0, 0.0], [1.0, 1.0]]
        self.dataset = {
            "version": 1, "document": "sample.txt",
            "cleaned_sha256": hashlib.sha256(b"evidence").hexdigest(), "cases": self.cases,
        }

    def read_dataset(self, dataset):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            return load_evaluation_dataset(path)

    def test_valid_dataset_and_document_are_accepted(self):
        self.assertEqual(self.read_dataset(self.dataset), self.dataset)
        validate_evaluation_document(self.dataset, "evidence")

    def test_empty_cases_unknown_version_and_missing_fields_are_rejected(self):
        for replacement in ({"cases": []}, {"version": 2}, {"document": ""}, {"cleaned_sha256": None}):
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    self.read_dataset(dict(self.dataset, **replacement))

    def test_duplicate_ids_and_invalid_case_records_are_rejected(self):
        for cases in ([self.cases[0], self.cases[0]], [None], [{"id": "one", "question": " "}]):
            with self.subTest(cases=cases):
                with self.assertRaises(ValueError):
                    self.read_dataset(dict(self.dataset, cases=cases))

    def test_invalid_evidence_values_are_rejected(self):
        for evidence in (None, "evidence", [" "], [123]):
            with self.subTest(evidence=evidence):
                cases = [dict(self.cases[0], evidence=evidence)]
                with self.assertRaises(ValueError):
                    self.read_dataset(dict(self.dataset, cases=cases))

    def test_changed_document_and_missing_quotes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "document changed"):
            validate_evaluation_document(self.dataset, "edited evidence")
        self.cases[0]["evidence"] = ["not present"]
        with self.assertRaisesRegex(ValueError, "evidence missing"):
            validate_evaluation_document(self.dataset, "evidence")

    def test_top_k_changes_hit_and_reciprocal_rank(self):
        missed = evaluate_retrieval(self.cases, self.chunks, self.vectors, [[1.0, 0.0]], top_k=1)
        found = evaluate_retrieval(self.cases, self.chunks, self.vectors, [[1.0, 0.0]], top_k=2)
        self.assertEqual(missed["hit_rate"], 0.0)
        self.assertEqual(missed["mrr"], 0.0)
        self.assertEqual(found["hit_rate"], 1.0)
        self.assertEqual(found["mrr"], 0.5)
        self.assertEqual(found["rows"][0]["relevant_indices"], [1])

    def test_unanswerable_cases_are_excluded_from_metric_denominator(self):
        cases = self.cases + [{"id": "negative", "question": "Absent?", "evidence": []}]
        report = evaluate_retrieval(cases, self.chunks, self.vectors, [[1.0, 1.0], [1.0, 0.0]])
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertEqual(report["answerable_count"], 1)
        self.assertEqual(report["unanswerable_count"], 1)
        self.assertIsNone(report["rows"][1]["hit"])
        self.assertIsNone(report["rows"][1]["reciprocal_rank"])
        self.assertTrue(report["rows"][1]["retrieved"])

    def test_all_unanswerable_cases_have_no_accuracy_metrics(self):
        cases = [dict(self.cases[0], evidence=[])]
        report = evaluate_retrieval(cases, self.chunks, self.vectors, [[1.0, 0.0]])
        self.assertIsNone(report["hit_rate"])
        self.assertIsNone(report["mrr"])

    def test_missing_queries_candidates_and_invalid_vectors_fail(self):
        for cases, candidates, queries in (
            ([], self.vectors, []),
            (self.cases, self.vectors, []),
            (self.cases, self.vectors[:1], [[1.0, 0.0]]),
            (self.cases, self.vectors, [[0.0, 0.0]]),
        ):
            with self.subTest(cases=len(cases), candidates=len(candidates), queries=queries):
                with self.assertRaises(ValueError):
                    evaluate_retrieval(cases, self.chunks, candidates, queries)

    def test_evidence_lost_at_chunk_boundaries_is_a_miss(self):
        report = evaluate_retrieval(self.cases, [{"index": 0, "text": "evid"}], [[1.0, 0.0]], [[1.0, 0.0]])
        self.assertEqual(report["hit_rate"], 0.0)

    def test_cli_uses_cached_model_and_prints_every_case(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["evaluate.py"]), \
             patch.object(evaluate.sys, "version_info", (3, 12)), \
             patch.object(evaluate, "load_local_model") as loader, \
             patch.object(evaluate, "embed_texts_locally", return_value=[[1.0, 0.0]] * 10), \
             redirect_stdout(output):
            evaluate.main()
        self.assertNotIn("allow_download", loader.call_args.kwargs)
        self.assertEqual(output.getvalue().count("  Retrieved:"), 6)
        self.assertIn("Answerable cases: 4", output.getvalue())
        self.assertIn("Unanswerable cases reported separately: 2", output.getvalue())
        self.assertIn("no answer generation", output.getvalue())
