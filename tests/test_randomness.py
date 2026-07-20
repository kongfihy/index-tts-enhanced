import unittest
from unittest.mock import patch

from indextts.utils.randomness import (
    MAX_INFERENCE_SEED,
    apply_inference_seed,
    candidate_seed_sequence,
    normalize_candidate_count,
    normalize_inference_seed,
)


class RandomnessTests(unittest.TestCase):
    def test_fixed_seed_is_preserved(self):
        self.assertEqual(normalize_inference_seed("123"), 123)
        self.assertEqual(normalize_inference_seed(0), 0)

    def test_negative_seed_requests_a_random_value(self):
        with patch("indextts.utils.randomness.secrets.randbelow", return_value=456):
            self.assertEqual(normalize_inference_seed(-1), 456)

    def test_seed_range_is_validated(self):
        with self.assertRaisesRegex(ValueError, "不能大于"):
            normalize_inference_seed(MAX_INFERENCE_SEED + 1)
        with self.assertRaisesRegex(ValueError, "不能小于"):
            normalize_inference_seed(-1, randomize_negative=False)

    def test_candidate_sequence_is_stable_and_wraps(self):
        self.assertEqual(candidate_seed_sequence(100, 3), [100, 101, 102])
        self.assertEqual(
            candidate_seed_sequence(MAX_INFERENCE_SEED, 2),
            [MAX_INFERENCE_SEED, 0],
        )


    def test_applying_same_seed_repeats_python_numpy_and_torch_rng(self):
        import random
        import numpy as np
        import torch

        apply_inference_seed(1234)
        first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
        apply_inference_seed(1234)
        second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

        self.assertEqual(first, second)

    def test_candidate_count_is_bounded(self):
        self.assertEqual(normalize_candidate_count(3), 3)
        with self.assertRaisesRegex(ValueError, "1 到 3"):
            normalize_candidate_count(4)


if __name__ == "__main__":
    unittest.main()
