"""Check ranking, provenance, complete inputs, and command-line selection."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import sys
import unittest
from unittest.mock import patch

import embedding_demo
from rag.retrieval import rank_chunks


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"id": str(index), "index": index, "text": text,
             "source": "notes.txt", "start": index * 10, "end": index * 10 + 10}
            for index, text in enumerate(("unrelated", "best passage", "partial match"))
        ]
        self.vectors = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.query = [1.0, 0.0]

    def test_top_k_returns_highest_scores_with_complete_source_metadata(self):
        matches = rank_chunks(self.chunks, self.vectors, self.query, top_k=2)
        self.assertEqual([chunk["index"] for chunk, score in matches], [1, 2])
        self.assertEqual(matches[0], (self.chunks[1], 1.0))
        self.assertAlmostEqual(matches[1][1], 0.7071067811865475)

    def test_top_one_and_limit_larger_than_collection(self):
        self.assertEqual(len(rank_chunks(self.chunks, self.vectors, self.query, 1)), 1)
        self.assertEqual(len(rank_chunks(self.chunks, self.vectors, self.query, 10)), 3)

    def test_tied_scores_keep_original_order(self):
        vectors = [[1.0, 0.0]] * 3
        matches = rank_chunks(self.chunks, vectors, self.query, 3)
        self.assertEqual([chunk["index"] for chunk, score in matches], [0, 1, 2])

    def test_negative_scores_are_ranked_without_claiming_relevance(self):
        matches = rank_chunks(self.chunks[:2], [[-1.0, 0.0], [-1.0, 1.0]], self.query)
        self.assertEqual([chunk["index"] for chunk, score in matches], [1, 0])
        self.assertLess(matches[0][1], 0)

    def test_empty_collection_returns_no_passages(self):
        self.assertEqual(rank_chunks([], [], self.query), [])

    def test_invalid_limits_are_rejected(self):
        for top_k in (0, -1, 1.5, "2", True, None):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    rank_chunks(self.chunks, self.vectors, self.query, top_k)

    def test_missing_or_extra_embeddings_are_rejected(self):
        for vectors in (self.vectors[:-1], self.vectors + [[1.0, 0.0]]):
            with self.subTest(count=len(vectors)):
                with self.assertRaisesRegex(ValueError, "exactly one embedding"):
                    rank_chunks(self.chunks, vectors, self.query)

    def test_invalid_query_is_rejected_even_for_empty_collection(self):
        for query in ([], [0.0, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]):
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    rank_chunks([], [], query)

    def test_invalid_candidate_is_rejected_even_beyond_requested_limit(self):
        for invalid in ([0.0, 0.0], [1.0], [float("nan"), 0.0], [float("inf"), 0.0]):
            with self.subTest(vector=invalid):
                with self.assertRaises(ValueError):
                    rank_chunks(self.chunks, self.vectors[:2] + [invalid], self.query, 1)

    def test_inputs_and_returned_metadata_are_independent(self):
        original = copy.deepcopy((self.chunks, self.vectors, self.query))
        matches = rank_chunks(self.chunks, self.vectors, self.query)
        matches[0][0]["text"] = "changed result"
        self.assertEqual((self.chunks, self.vectors, self.query), original)

    def test_cli_prints_selected_passage_and_source(self):
        output = io.StringIO()
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        with patch.object(sys, "argv", ["embedding_demo.py", "--top-k", "1"]), \
             patch.object(embedding_demo.sys, "version_info", (3, 12)), \
             patch.object(embedding_demo, "load_local_model"), \
             patch.object(embedding_demo, "embed_texts_locally", return_value=vectors), \
             redirect_stdout(output):
            embedding_demo.main()
        retrieved = output.getvalue().split("Retrieved passages:")[1]
        self.assertIn("1 of 4 (top_k=1)", retrieved)
        self.assertIn("Rank 1: chunk 1 | cosine=1.0000", retrieved)
        self.assertIn("Source:", retrieved)
        self.assertIn("Offsets: [100:220]", retrieved)
        self.assertNotIn("Rank 2:", retrieved)

    def test_cli_rejects_invalid_limit_before_loading_model(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["embedding_demo.py", "--top-k", value]), \
                     patch.object(embedding_demo, "load_local_model") as loader, \
                     redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit) as caught:
                    embedding_demo.main()
                self.assertEqual(caught.exception.code, 2)
                loader.assert_not_called()
