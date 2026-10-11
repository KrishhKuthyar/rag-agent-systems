"""Check that prose cleanup preserves evidence and document structure."""

import unittest

from rag.preprocessing import clean_text


class CleanTextTests(unittest.TestCase):
    def test_preserves_unicode_and_paragraph_boundaries(self):
        original = "  Notes about café.\r\n\r\n\r\n  Second\tparagraph: 東京.  "
        self.assertEqual(
            clean_text(original),
            "Notes about café.\n\nSecond paragraph: 東京.",
        )

    def test_preserves_single_line_breaks(self):
        self.assertEqual(clean_text("First line.\rSecond line."), "First line.\nSecond line.")

    def test_preserves_words_punctuation_and_nonbreaking_spaces(self):
        self.assertEqual(
            clean_text("  Price:  1,000.50!\tKeep A\u00a0B.  "),
            "Price: 1,000.50! Keep A\u00a0B.",
        )

    def test_empty_or_whitespace_only_text_is_empty(self):
        for text in ("", " \t\r\n\n  "):
            with self.subTest(text=text):
                self.assertEqual(clean_text(text), "")

    def test_cleaning_is_idempotent(self):
        original = "\n  First   paragraph.\n\n\n\tSecond paragraph.\n"
        cleaned = clean_text(original)
        self.assertEqual(clean_text(cleaned), cleaned)
