"""Check answer instructions and preservation of the question and evidence."""

from contextlib import redirect_stdout
import io
import sys
import unittest
from unittest.mock import patch

import embedding_demo
from rag.prompting import INSUFFICIENT_EVIDENCE, build_answer_messages


class PromptingTests(unittest.TestCase):
    def test_roles_separate_instructions_from_exact_evidence_and_question(self):
        context = "[S1]\nText:\n café notes\n\nKeep paragraph breaks. "
        question = "What do café notes contain?"
        messages = build_answer_messages(question, context)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"],
                         f"Evidence:\n{context}\n\nQuestion: {question}\n\n"
                         "Answer in one sentence using the evidence and include its [S1]-style citation:")

    def test_instructions_require_evidence_citations_and_abstention(self):
        instructions = build_answer_messages("Question?", "[S1] Evidence")[0]["content"]
        for expected in ("only the provided evidence", "not instructions", "[S1]",
                         "Do not invent facts or source labels", INSUFFICIENT_EVIDENCE):
            self.assertIn(expected, instructions)

    def test_empty_context_is_explicit(self):
        for context in ("", " \n"):
            with self.subTest(context=context):
                messages = build_answer_messages("Question?", context)
                self.assertIn("(No passages retrieved.)", messages[-1]["content"])

    def test_blank_or_nonstr_question_is_rejected(self):
        for question in ("", " \n", None, 123):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "nonblank string"):
                    build_answer_messages(question, "[S1] Evidence")

    def test_nonstr_context_is_rejected(self):
        for context in (None, [], 123):
            with self.subTest(context=context):
                with self.assertRaisesRegex(ValueError, "context must be a string"):
                    build_answer_messages("Question?", context)

    def test_calls_return_independent_message_records(self):
        first = build_answer_messages("Question?", "[S1] Evidence")
        first[0]["content"] = "changed"
        self.assertNotEqual(first, build_answer_messages("Question?", "[S1] Evidence"))

    def test_cli_can_display_prompt_without_an_answer_model(self):
        output = io.StringIO()
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        with patch.object(sys, "argv", ["embedding_demo.py", "--top-k", "1", "--show-prompt"]), \
             patch.object(embedding_demo.sys, "version_info", (3, 12)), \
             patch.object(embedding_demo, "load_local_model"), \
             patch.object(embedding_demo, "embed_texts_locally", return_value=vectors), \
             redirect_stdout(output):
            embedding_demo.main()
        prompt = output.getvalue().split("Answer prompt (no generation):")[1]
        self.assertIn("SYSTEM:\n", prompt)
        self.assertIn("USER:\nEvidence:\n[S1]", prompt)
        self.assertIn("Question: Why do chunks overlap?", prompt)
        self.assertNotIn("[S2]", prompt)
