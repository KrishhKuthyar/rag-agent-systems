"""Check embedding input mapping and rejection of incomplete or invalid results."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rag.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, embed_texts


def vector_at(axis):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[axis] = 1.0
    return vector


def make_client(entries, model=EMBEDDING_MODEL):
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(model=model, data=entries)
    return client


def entry(index, vector):
    return SimpleNamespace(index=index, embedding=vector)


class EmbedTextsTests(unittest.TestCase):
    def test_request_uses_one_model_and_restores_input_order(self):
        client = make_client([entry(1, vector_at(1)), entry(0, vector_at(0))])
        texts = [" first passage ", "Why do chunks overlap?"]
        vectors = embed_texts(texts, client)
        self.assertEqual(vectors, [vector_at(0), vector_at(1)])
        client.embeddings.create.assert_called_once_with(
            model=EMBEDDING_MODEL,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )

    def test_empty_batch_does_not_call_api(self):
        client = Mock()
        self.assertEqual(embed_texts([], client), [])
        client.embeddings.create.assert_not_called()

    def test_single_string_is_not_treated_as_a_batch_of_characters(self):
        for texts in ("passage", b"passage"):
            with self.subTest(texts=texts):
                client = Mock()
                with self.assertRaisesRegex(ValueError, "sequence of strings"):
                    embed_texts(texts, client)
                client.embeddings.create.assert_not_called()

    def test_blank_and_nonstring_items_are_rejected_before_api_call(self):
        for text in ("", " \n\t", None, 123):
            with self.subTest(text=text):
                client = Mock()
                with self.assertRaisesRegex(ValueError, "nonblank string"):
                    embed_texts(["valid", text], client)
                client.embeddings.create.assert_not_called()

    def test_partial_response_is_rejected(self):
        client = make_client([entry(0, vector_at(0))])
        with self.assertRaisesRegex(ValueError, "one embedding per input"):
            embed_texts(["first", "second"], client)

    def test_duplicate_missing_and_noninteger_indexes_are_rejected(self):
        for indexes in ((0, 0), (0, 2), (-1, 0), (0, 1.0), (0, True)):
            with self.subTest(indexes=indexes):
                client = make_client([entry(index, vector_at(0)) for index in indexes])
                with self.assertRaisesRegex(ValueError, "indexes"):
                    embed_texts(["first", "second"], client)

    def test_response_model_must_match_requested_model(self):
        client = make_client([entry(0, vector_at(0))], model="another-model")
        with self.assertRaisesRegex(ValueError, "model"):
            embed_texts(["passage"], client)

    def test_wrong_vector_size_or_format_is_rejected(self):
        for vector in ([], [1.0, 0.0], None, "x" * EMBEDDING_DIMENSIONS):
            with self.subTest(vector_type=type(vector).__name__):
                client = make_client([entry(0, vector)])
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    embed_texts(["passage"], client)

    def test_nonfinite_or_nonnumeric_coordinates_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), "invalid"):
            with self.subTest(value=value):
                vector = vector_at(0)
                vector[-1] = value
                with self.assertRaisesRegex(ValueError, "finite numbers"):
                    embed_texts(["passage"], make_client([entry(0, vector)]))

    def test_zero_vector_is_rejected(self):
        client = make_client([entry(0, [0.0] * EMBEDDING_DIMENSIONS)])
        with self.assertRaisesRegex(ValueError, "zero embeddings"):
            embed_texts(["passage"], client)

    def test_client_failure_is_not_replaced_with_fake_vectors(self):
        client = Mock()
        client.embeddings.create.side_effect = RuntimeError("service unavailable")
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            embed_texts(["passage"], client)
        self.assertEqual(client.embeddings.create.call_count, 1)

    def test_inputs_and_response_vectors_are_not_modified(self):
        texts = ["passage"]
        response_vector = vector_at(0)
        client = make_client([entry(0, response_vector)])
        result = embed_texts(texts, client)
        result[0][0] = 99.0
        self.assertEqual(texts, ["passage"])
        self.assertEqual(response_vector, vector_at(0))
