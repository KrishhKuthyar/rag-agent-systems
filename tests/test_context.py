"""Check evidence labels, exact passage text, provenance, and runner wiring."""

from contextlib import redirect_stdout
import copy
import io
import sys
import unittest
from unittest.mock import patch

import embedding_demo
from rag.context import build_context


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "id": "overlap-id", "index": 7, "text": "Overlap preserves context.",
            "source": "notes.txt", "start": 100, "end": 126,
        }

    def test_empty_selection_returns_empty_context(self):
        self.assertEqual(build_context([]), "")

    def test_labels_follow_retrieval_order_and_retain_provenance(self):
        second = dict(self.chunk, id="second-id", index=0, source="other.txt")
        context = build_context([(self.chunk, 0.8), (second, 0.7)])
        first_block, second_block = context.split("\n\n")
        self.assertEqual(first_block,
                         "[S1]\nSource: notes.txt\nChunk ID: overlap-id\n"
                         "Offsets: [100:126]\nText:\nOverlap preserves context.")
        self.assertTrue(second_block.startswith("[S2]\nSource: other.txt\n"))
        self.assertIn("Chunk ID: second-id", second_block)

    def test_unicode_whitespace_and_paragraphs_are_preserved_exactly(self):
        text = "  café notes\n\nKeep this spacing.\n"
        chunk = dict(self.chunk, text=text, end=100 + len(text), source="café.txt")
        context = build_context([(chunk, 0.5)])
        self.assertEqual(context.split("Text:\n", 1)[1], text)
        self.assertIn("Source: café.txt", context)

    def test_missing_blank_or_nonstr_metadata_is_rejected(self):
        for field in ("text", "source", "id"):
            for value in (None, "", " \n", 123):
                with self.subTest(field=field, value=value):
                    chunk = dict(self.chunk)
                    if value is None:
                        del chunk[field]
                    else:
                        chunk[field] = value
                    with self.assertRaisesRegex(ValueError, f"nonblank {field}"):
                        build_context([(chunk, 0.5)])

    def test_invalid_offsets_are_rejected(self):
        for start, end in ((-1, 25), (100, 100), (101, 100), (True, 27),
                           (100, False), (100.0, 126), (100, "126"), (None, 126)):
            with self.subTest(start=start, end=end):
                chunk = dict(self.chunk, start=start, end=end)
                with self.assertRaisesRegex(ValueError, "valid character offsets"):
                    build_context([(chunk, 0.5)])

    def test_inconsistent_offsets_reject_entire_selection(self):
        invalid = dict(self.chunk, end=125)
        with self.assertRaisesRegex(ValueError, "passage 2 offsets do not match"):
            build_context([(self.chunk, 0.8), (invalid, 0.7)])

    def test_input_records_are_unchanged(self):
        matches = [(self.chunk, 0.8)]
        original = copy.deepcopy(matches)
        build_context(matches)
        self.assertEqual(matches, original)

    def test_runner_formats_only_selected_passages(self):
        output = io.StringIO()
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        with patch.object(sys, "argv", ["embedding_demo.py", "--top-k", "1"]), \
             patch.object(embedding_demo.sys, "version_info", (3, 12)), \
             patch.object(embedding_demo, "load_local_model"), \
             patch.object(embedding_demo, "embed_texts_locally", return_value=vectors), \
             redirect_stdout(output):
            embedding_demo.main()
        context = output.getvalue().split("Context for answer generation:\n")[1]
        self.assertTrue(context.startswith("[S1]\nSource:"))
        self.assertIn("Offsets: [100:220]", context)
        self.assertIn("Chunk overlap repeats some text", context)
        self.assertNotIn("[S2]", context)
        self.assertNotIn("RAG study notes", context)
