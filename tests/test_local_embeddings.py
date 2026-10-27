"""Check local embedding integrity, token limits, and offline defaults."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import embedding_demo
from rag.local_embeddings import (
    LOCAL_DIMENSIONS,
    LOCAL_MODEL,
    LOCAL_MODEL_REVISION,
    embed_texts_locally,
    load_local_model,
)


def vector_at(axis):
    vector = [0.0] * LOCAL_DIMENSIONS
    vector[axis] = 1.0
    return vector


def make_model(count=1, vectors=None):
    if vectors is None:
        vectors = [vector_at(index) for index in range(count)]
    model = Mock()
    model.max_seq_length = 256
    model.get_embedding_dimension.return_value = LOCAL_DIMENSIONS
    model.tokenizer.return_value = {"input_ids": [[101, 102] for _ in range(count)]}
    model.encode.return_value = SimpleNamespace(tolist=lambda: vectors)
    return model


class LocalEmbeddingTests(unittest.TestCase):
    def test_encoding_preserves_input_order_and_requests_normalization(self):
        texts = ["café notes", "Why do chunks overlap?"]
        model = make_model(count=2)
        self.assertEqual(embed_texts_locally(texts, model), [vector_at(0), vector_at(1)])
        model.tokenizer.assert_called_once_with(
            texts, truncation=False, add_special_tokens=True
        )
        model.encode.assert_called_once_with(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def test_empty_batch_does_not_run_tokenizer_or_model(self):
        model = make_model()
        self.assertEqual(embed_texts_locally([], model), [])
        model.tokenizer.assert_not_called()
        model.encode.assert_not_called()

    def test_single_string_and_blank_items_are_rejected_before_inference(self):
        for texts in ("passage", b"passage", [""], [" \n"], [None]):
            with self.subTest(texts=texts):
                model = make_model()
                with self.assertRaises(ValueError):
                    embed_texts_locally(texts, model)
                model.tokenizer.assert_not_called()
                model.encode.assert_not_called()

    def test_oversized_input_is_rejected_instead_of_truncated(self):
        model = make_model()
        model.tokenizer.return_value = {"input_ids": [[1] * 257]}
        with self.assertRaisesRegex(ValueError, "input 0 has 257 tokens"):
            embed_texts_locally(["long passage"], model)
        model.encode.assert_not_called()

    def test_input_at_token_limit_is_accepted(self):
        model = make_model()
        model.tokenizer.return_value = {"input_ids": [[1] * 256]}
        self.assertEqual(embed_texts_locally(["passage"], model), [vector_at(0)])

    def test_missing_tokenizer_sequences_are_rejected(self):
        model = make_model()
        model.tokenizer.return_value = {"input_ids": []}
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            embed_texts_locally(["passage"], model)
        model.encode.assert_not_called()

    def test_partial_embedding_batch_is_rejected(self):
        model = make_model(count=2, vectors=[vector_at(0)])
        with self.assertRaisesRegex(ValueError, "one embedding per input"):
            embed_texts_locally(["first", "second"], model)

    def test_wrong_vector_dimensions_or_format_are_rejected(self):
        for vector in ([], [1.0] * 383, [1.0] * 385, None):
            with self.subTest(vector_type=type(vector).__name__):
                model = make_model(vectors=[vector])
                with self.assertRaisesRegex(ValueError, "384 dimensions"):
                    embed_texts_locally(["passage"], model)

    def test_nonfinite_or_nonnumeric_coordinates_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), "invalid"):
            with self.subTest(value=value):
                vector = vector_at(0)
                vector[-1] = value
                model = make_model(vectors=[vector])
                with self.assertRaisesRegex(ValueError, "finite numbers"):
                    embed_texts_locally(["passage"], model)

    def test_zero_vector_is_rejected(self):
        model = make_model(vectors=[[0.0] * LOCAL_DIMENSIONS])
        with self.assertRaisesRegex(ValueError, "zero embeddings"):
            embed_texts_locally(["passage"], model)

    def test_model_failures_propagate_without_replacement_vectors(self):
        model = make_model()
        model.encode.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            embed_texts_locally(["passage"], model)

    def test_loader_uses_pinned_cpu_model_and_cached_files_by_default(self):
        constructor = Mock(return_value=make_model())
        with TemporaryDirectory() as folder:
            cache = Path(folder) / "models"
            with patch.dict(
                sys.modules,
                {"sentence_transformers": SimpleNamespace(SentenceTransformer=constructor)},
            ):
                load_local_model(cache)
            self.assertTrue(cache.is_dir())
            constructor.assert_called_once_with(
                LOCAL_MODEL,
                revision=LOCAL_MODEL_REVISION,
                cache_folder=str(cache),
                device="cpu",
                local_files_only=True,
                token=False,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )

    def test_loader_rejects_an_unexpected_model_dimension(self):
        model = make_model()
        model.get_embedding_dimension.return_value = 1536
        with TemporaryDirectory() as folder:
            with patch.dict(
                sys.modules,
                {"sentence_transformers": SimpleNamespace(SentenceTransformer=Mock(return_value=model))},
            ):
                with self.assertRaisesRegex(ValueError, "384 dimensions"):
                    load_local_model(Path(folder))

    def test_returned_vectors_do_not_share_mutable_input_or_response_lists(self):
        texts = ["passage"]
        response_vector = vector_at(0)
        result = embed_texts_locally(texts, make_model(vectors=[response_vector]))
        result[0][0] = 99.0
        self.assertEqual(texts, ["passage"])
        self.assertEqual(response_vector, vector_at(0))

    def test_default_runner_works_without_api_credentials_or_api_libraries(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
             patch.dict(sys.modules, {"openai": None, "dotenv": None}), \
             patch.object(embedding_demo.sys, "version_info", (3, 12)), \
             patch.object(sys, "argv", ["embedding_demo.py"]), \
             patch.object(embedding_demo, "load_local_model", return_value=make_model(5)) as loader, \
             redirect_stdout(output):
            embedding_demo.main()
        self.assertFalse(loader.call_args.kwargs["allow_download"])
        self.assertIn("Mode: cached offline inference", output.getvalue())
        self.assertIn("Accepted local embeddings: 5", output.getvalue())
        self.assertEqual(output.getvalue().count("Cosine similarity to query:"), 4)
