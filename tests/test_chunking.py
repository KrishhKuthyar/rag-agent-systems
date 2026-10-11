"""Check chunk coverage, overlap boundaries, and source provenance."""

import unittest

from rag.chunking import chunk_document


class ChunkDocumentTests(unittest.TestCase):
    def test_known_overlap_boundaries_and_source_metadata(self):
        document = {"text": "abcdefghij", "source": "notes.txt"}
        chunks = chunk_document(document, chunk_size=6, overlap=2)

        self.assertEqual([chunk["text"] for chunk in chunks], ["abcdef", "efghij"])
        self.assertEqual(
            [(chunk["start"], chunk["end"]) for chunk in chunks],
            [(0, 6), (4, 10)],
        )
        self.assertEqual([chunk["index"] for chunk in chunks], [0, 1])
        self.assertTrue(all(chunk["source"] == "notes.txt" for chunk in chunks))

    def test_zero_overlap_covers_all_text_including_short_final_chunk(self):
        document = {"text": "abcdefghij", "source": "notes.txt"}
        chunks = chunk_document(document, chunk_size=4, overlap=0)

        self.assertEqual([chunk["text"] for chunk in chunks], ["abcd", "efgh", "ij"])
        self.assertEqual("".join(chunk["text"] for chunk in chunks), document["text"])

    def test_text_at_or_below_chunk_size_produces_one_chunk(self):
        for text in ("abc", "abcd"):
            with self.subTest(text=text):
                chunks = chunk_document(
                    {"text": text, "source": "notes.txt"},
                    chunk_size=4,
                    overlap=2,
                )
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0]["text"], text)
                self.assertEqual(chunks[0]["end"], len(text))

    def test_maximum_valid_overlap_advances_without_duplicate_tail(self):
        chunks = chunk_document(
            {"text": "abcdef", "source": "notes.txt"},
            chunk_size=4,
            overlap=3,
        )
        self.assertEqual([chunk["text"] for chunk in chunks], ["abcd", "bcde", "cdef"])

    def test_empty_document_produces_no_chunks(self):
        self.assertEqual(chunk_document({"text": "", "source": "empty.txt"}), [])

    def test_invalid_chunk_size_is_rejected_even_for_empty_text(self):
        for chunk_size in (0, -1):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    chunk_document(
                        {"text": "", "source": "notes.txt"},
                        chunk_size=chunk_size,
                        overlap=0,
                    )

    def test_invalid_overlap_is_rejected_even_for_empty_text(self):
        for overlap in (-1, 4, 5):
            with self.subTest(overlap=overlap):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    chunk_document(
                        {"text": "", "source": "notes.txt"},
                        chunk_size=4,
                        overlap=overlap,
                    )

    def test_unicode_and_paragraphs_can_be_reconstructed_without_loss(self):
        text = "A café note.\n\nNext paragraph: 東京."
        chunks = chunk_document(
            {"text": text, "source": "café.txt"},
            chunk_size=7,
            overlap=2,
        )

        reconstructed = chunks[0]["text"]
        for chunk in chunks[1:]:
            reconstructed += chunk["text"][2:]
        self.assertEqual(reconstructed, text)
        for chunk in chunks:
            self.assertEqual(chunk["text"], text[chunk["start"]:chunk["end"]])
            self.assertLessEqual(len(chunk["text"]), 7)

    def test_original_document_is_not_modified(self):
        document = {"text": "abcdefghij", "source": "notes.txt"}
        original = document.copy()
        chunk_document(document, chunk_size=6, overlap=2)
        self.assertEqual(document, original)
