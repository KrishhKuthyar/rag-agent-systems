"""Check cosine geometry, invalid inputs, and numerical stability."""

import unittest

from rag.similarity import cosine_similarity


class CosineSimilarityTests(unittest.TestCase):
    def test_known_geometric_scores(self):
        examples = (
            ([1, 0], [2, 0], 1.0),
            ([1, 0], [1, 1], 0.7071067811865475),
            ([1, 0], [0, 1], 0.0),
            ([1, 0], [-1, 0], -1.0),
            ([1, 2], [2, 1], 0.8),
        )
        for a, b, expected in examples:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(cosine_similarity(a, b), expected, places=12)

    def test_similarity_is_symmetric(self):
        a = [1, -2, 3]
        b = [4, 5, -6]
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a))

    def test_positive_rescaling_preserves_similarity(self):
        a = [1, -2, 3]
        b = [4, 5, -6]
        self.assertAlmostEqual(
            cosine_similarity(a, b),
            cosine_similarity([value * 7.5 for value in a], [value * 0.25 for value in b]),
        )

    def test_large_and_small_finite_coordinates_keep_the_expected_direction(self):
        for magnitude in (1e308, 1e-308, 5e-324):
            with self.subTest(magnitude=magnitude):
                self.assertAlmostEqual(
                    cosine_similarity([magnitude, 0], [magnitude, magnitude]),
                    0.7071067811865475,
                )

    def test_empty_vectors_are_rejected(self):
        for a, b in (([], []), ([], [1]), ([1], [])):
            with self.subTest(a=a, b=b):
                with self.assertRaisesRegex(ValueError, "empty"):
                    cosine_similarity(a, b)

    def test_unequal_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal dimensions"):
            cosine_similarity([1], [1, 2])

    def test_zero_vectors_are_rejected(self):
        for a, b in (([0, 0], [1, 0]), ([1, 0], [0, 0]), ([0, 0], [0, 0])):
            with self.subTest(a=a, b=b):
                with self.assertRaisesRegex(ValueError, "zero vectors"):
                    cosine_similarity(a, b)

    def test_nonfinite_coordinates_are_rejected_in_either_vector(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            for a, b in (([invalid, 0], [1, 0]), ([1, 0], [invalid, 0])):
                with self.subTest(a=a, b=b):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        cosine_similarity(a, b)

    def test_input_vectors_are_not_modified(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        original_a = a.copy()
        original_b = b.copy()
        cosine_similarity(a, b)
        self.assertEqual(a, original_a)
        self.assertEqual(b, original_b)

    def test_tuples_and_lists_are_supported(self):
        self.assertEqual(cosine_similarity((1, 0), [2, 0]), 1.0)

    def test_rounding_stays_within_the_cosine_range(self):
        a = [0.1, 0.2, 0.3]
        for b in (a, [-value for value in a]):
            with self.subTest(b=b):
                score = cosine_similarity(a, b)
                self.assertGreaterEqual(score, -1.0)
                self.assertLessEqual(score, 1.0)
