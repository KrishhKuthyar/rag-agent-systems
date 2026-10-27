"""Check repeatable passage identity across sources, changes, and processes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from rag.chunking import chunk_document


class ChunkIdTests(unittest.TestCase):
    def test_repeated_ingestion_preserves_version_one_ids(self):
        document = {"text": "abcdefghij", "source": "notes.txt"}
        first = chunk_document(document, chunk_size=6, overlap=2)
        second = chunk_document(document, chunk_size=6, overlap=2)

        self.assertEqual(first, second)
        # Pin the identity format so serialization changes cannot silently rename chunks.
        self.assertEqual(
            [chunk["id"] for chunk in first],
            [
                "81d339606e747275c108694cdf3b946016a58f573c983432af588d827e712521",
                "8e9cdfa7241e85beafa0a33b6872952effd6482d79da308c39b680224e589fee",
            ],
        )
        for chunk in first:
            self.assertRegex(chunk["id"], r"^[0-9a-f]{64}$")

    def test_identical_text_from_different_sources_has_different_ids(self):
        first = chunk_document(
            {"text": "abcdefghij", "source": "notes.txt"}, chunk_size=6, overlap=2
        )
        second = chunk_document(
            {"text": "abcdefghij", "source": "other.txt"}, chunk_size=6, overlap=2
        )
        first_ids = {chunk["id"] for chunk in first}
        second_ids = {chunk["id"] for chunk in second}
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_repeated_passages_at_different_positions_have_different_ids(self):
        chunks = chunk_document(
            {"text": "abcabc", "source": "notes.txt"}, chunk_size=3, overlap=0
        )
        self.assertEqual([chunk["text"] for chunk in chunks], ["abc", "abc"])
        self.assertNotEqual(chunks[0]["id"], chunks[1]["id"])

    def test_edited_passage_changes_id_while_unchanged_passage_keeps_id(self):
        original = chunk_document(
            {"text": "abcdef", "source": "notes.txt"}, chunk_size=3, overlap=0
        )
        edited = chunk_document(
            {"text": "abcXef", "source": "notes.txt"}, chunk_size=3, overlap=0
        )
        self.assertEqual(original[0]["id"], edited[0]["id"])
        self.assertNotEqual(original[1]["id"], edited[1]["id"])

    def test_identity_includes_text_beyond_the_first_fifty_characters(self):
        original = chunk_document(
            {"text": "a" * 60 + "X", "source": "notes.txt"}
        )
        edited = chunk_document(
            {"text": "a" * 60 + "Y", "source": "notes.txt"}
        )
        self.assertNotEqual(original[0]["id"], edited[0]["id"])

    def test_same_passage_keeps_id_when_only_its_sequence_index_changes(self):
        document = {"text": "abcabcabc", "source": "notes.txt"}
        first = chunk_document(document, chunk_size=3, overlap=0)
        second = chunk_document(document, chunk_size=3, overlap=1)

        # Both last chunks cover [6:9], but they are numbered 2 and 3.
        self.assertNotEqual(first[-1]["index"], second[-1]["index"])
        self.assertEqual(first[-1]["id"], second[-1]["id"])

    def test_unicode_ids_are_repeatable_in_separate_python_processes(self):
        document = {"text": "café 東京", "source": "café-notes.txt"}
        expected = [
            chunk["id"]
            for chunk in chunk_document(document, chunk_size=4, overlap=1)
        ]
        code = (
            "import json\n"
            "from rag.chunking import chunk_document\n"
            f"chunks = chunk_document({document!r}, chunk_size=4, overlap=1)\n"
            "print(json.dumps([chunk['id'] for chunk in chunks]))\n"
        )
        project_root = Path(__file__).resolve().parent.parent
        for seed in ("1", "2"):
            with self.subTest(seed=seed):
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=project_root,
                    env={**os.environ, "PYTHONHASHSEED": seed},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(json.loads(result.stdout), expected)
