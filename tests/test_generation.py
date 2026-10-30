"""Check offline loading, bounded generation, source labels, and the RAG runner."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import rag_demo
from rag.generation import (
    ANSWER_MODEL, ANSWER_MODEL_REVISION, MAX_INPUT_TOKENS,
    check_answer_citations, generate_answer, load_answer_model,
)
from rag.prompting import INSUFFICIENT_EVIDENCE


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Question and evidence"}]
        self.tokenizer = Mock(eos_token_id=99)
        self.tokenizer.apply_chat_template.return_value = {
            "input_ids": SimpleNamespace(shape=(1, 5)), "attention_mask": "mask",
        }
        self.tokenizer.decode.return_value = "  An answer. [S1]  "
        self.model = Mock()
        self.model.config.max_position_embeddings = 32768
        self.model.generation_config.eos_token_id = [99]
        self.set_output([10, 11, 99])
        self.torch = SimpleNamespace(inference_mode=nullcontext, float32="float32")

    def set_output(self, tokens):
        row = MagicMock()
        row.__getitem__.return_value = SimpleNamespace(tolist=lambda: tokens)
        self.model.generate.return_value = [row]
        return row

    def run_generation(self, **kwargs):
        with patch.dict(sys.modules, {"torch": self.torch}):
            return generate_answer(self.messages, self.tokenizer, self.model, **kwargs)

    def test_loader_pins_both_components_and_defaults_to_cpu_cached_files(self):
        tokenizer_class = Mock()
        model_class = Mock()
        module = SimpleNamespace(AutoTokenizer=tokenizer_class, AutoModelForCausalLM=model_class)
        with TemporaryDirectory() as folder, patch.dict(sys.modules, {"torch": self.torch, "transformers": module}):
            tokenizer, model = load_answer_model(Path(folder))
        settings = {
            "revision": ANSWER_MODEL_REVISION, "cache_dir": folder,
            "local_files_only": True, "token": False, "trust_remote_code": False,
        }
        tokenizer_class.from_pretrained.assert_called_once_with(ANSWER_MODEL, **settings)
        model_class.from_pretrained.assert_called_once_with(
            ANSWER_MODEL, **settings, use_safetensors=True, dtype="float32")
        self.assertIs(tokenizer, tokenizer_class.from_pretrained.return_value)
        self.assertIs(model, model_class.from_pretrained.return_value)
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()

    def test_generation_uses_chat_template_and_decodes_only_new_tokens(self):
        row = self.set_output([10, 11, 99])
        self.assertEqual(self.run_generation(), "An answer. [S1]")
        self.tokenizer.apply_chat_template.assert_called_once_with(
            self.messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", truncation=False)
        row.__getitem__.assert_called_once_with(slice(5, None, None))
        self.tokenizer.decode.assert_called_once_with([10, 11, 99], skip_special_tokens=True)
        self.assertFalse(self.model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 128)

    def test_invalid_output_limits_or_empty_messages_fail_before_inference(self):
        for limit in (0, -1, 257, 1.5, True):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "between 1 and 256"):
                    self.run_generation(max_new_tokens=limit)
        self.messages = []
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.run_generation()
        self.model.generate.assert_not_called()

    def test_oversized_prompt_is_rejected_without_truncation(self):
        self.tokenizer.apply_chat_template.return_value["input_ids"].shape = (1, MAX_INPUT_TOKENS + 1)
        with self.assertRaisesRegex(ValueError, "prompt has"):
            self.run_generation()
        self.model.generate.assert_not_called()

    def test_context_window_includes_requested_answer_tokens(self):
        self.model.config.max_position_embeddings = 100
        with self.assertRaisesRegex(ValueError, "context window"):
            self.run_generation()
        self.model.generate.assert_not_called()

    def test_empty_answer_is_rejected(self):
        self.tokenizer.decode.return_value = " \n"
        with self.assertRaisesRegex(ValueError, "empty answer"):
            self.run_generation()

    def test_token_limit_without_stop_token_is_reported_as_incomplete(self):
        self.set_output([10, 11, 12])
        with self.assertRaisesRegex(ValueError, "before completion"):
            self.run_generation(max_new_tokens=3)
        self.tokenizer.decode.assert_not_called()

    def test_stop_token_at_limit_is_a_complete_answer(self):
        self.model.generation_config.eos_token_id = 99
        self.assertEqual(self.run_generation(max_new_tokens=3), "An answer. [S1]")

    def test_generation_failures_propagate(self):
        self.model.generate.side_effect = RuntimeError("generation failed")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.run_generation()

    def test_existing_labels_and_exact_abstention_are_accepted(self):
        self.assertIsNone(check_answer_citations("First [S2], then [S1].", 2))
        self.assertIsNone(check_answer_citations(INSUFFICIENT_EVIDENCE, 0))

    def test_missing_or_unavailable_citations_are_rejected(self):
        for answer in ("No citations.", "Fact [S3]", "Fact [S0]", "Fact [S01]", "Fact [S1] [Sfoo]"):
            with self.subTest(answer=answer):
                with self.assertRaises(ValueError):
                    check_answer_citations(answer, 2)

    def test_blank_answers_and_invalid_source_counts_are_rejected(self):
        for answer in ("", " \n", None):
            with self.subTest(answer=answer):
                with self.assertRaises(ValueError):
                    check_answer_citations(answer, 2)
        for count in (-1, True, 1.5):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    check_answer_citations("Fact [S1]", count)


class RagRunnerTests(unittest.TestCase):
    def test_default_runner_uses_only_local_models_and_displays_answer(self):
        output = io.StringIO()
        vectors = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
        with patch.object(sys, "argv", ["rag_demo.py", "--top-k", "1"]), \
             patch.object(rag_demo.sys, "version_info", (3, 12)), \
             patch.dict(sys.modules, {"openai": None, "dotenv": None}), \
             patch.object(rag_demo, "load_local_model") as embedding_loader, \
             patch.object(rag_demo, "embed_texts_locally", return_value=vectors), \
             patch.object(rag_demo, "load_answer_model", return_value=(Mock(), Mock())) as answer_loader, \
             patch.object(rag_demo, "generate_answer", return_value="Overlap preserves context. [S1]"), \
             redirect_stdout(output):
            rag_demo.main()
        self.assertFalse(embedding_loader.call_args.kwargs["allow_download"])
        self.assertFalse(answer_loader.call_args.kwargs["allow_download"])
        self.assertIn("Generated answer:\nOverlap preserves context. [S1]", output.getvalue())
        self.assertIn("Citation labels checked", output.getvalue())
        self.assertIn("Offsets: [100:220]", output.getvalue())

    def test_empty_document_abstains_without_loading_any_model(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["rag_demo.py"]), \
             patch.object(rag_demo.sys, "version_info", (3, 12)), \
             patch.object(rag_demo, "load_document", return_value={"text": "", "source": "empty.txt"}), \
             patch.object(rag_demo, "load_local_model") as embeddings, \
             patch.object(rag_demo, "load_answer_model") as answers, \
             redirect_stdout(output):
            rag_demo.main()
        embeddings.assert_not_called()
        answers.assert_not_called()
        self.assertIn(INSUFFICIENT_EVIDENCE, output.getvalue())

    def test_runner_reports_bad_citations_as_failure(self):
        output = io.StringIO()
        vectors = [[1.0, 0.0]] * 5
        with patch.object(sys, "argv", ["rag_demo.py"]), \
             patch.object(rag_demo.sys, "version_info", (3, 12)), \
             patch.object(rag_demo, "load_local_model"), \
             patch.object(rag_demo, "embed_texts_locally", return_value=vectors), \
             patch.object(rag_demo, "load_answer_model", return_value=(Mock(), Mock())), \
             patch.object(rag_demo, "generate_answer", return_value="Invented source [S9]"), \
             redirect_stdout(output), redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as caught:
            rag_demo.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("Generated answer:\nInvented source [S9]", output.getvalue())
        self.assertNotIn("Citation labels checked", output.getvalue())

    def test_runner_rejects_invalid_options_before_model_loading(self):
        for option, value in (("--query", " "), ("--top-k", "0"), ("--max-new-tokens", "0"), ("--max-new-tokens", "257")):
            with self.subTest(option=option, value=value):
                with patch.object(sys, "argv", ["rag_demo.py", option, value]), \
                     patch.object(rag_demo, "load_local_model") as loader, \
                     redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    rag_demo.main()
                self.assertEqual(caught.exception.code, 2)
                loader.assert_not_called()
