"""Arithmetic checks on invented inputs; no experiment or clinical data."""

from dataclasses import FrozenInstanceError
import math
import unittest

import numpy as np

from norm_core.support import (
    SupportState,
    adam_support_step,
    normalize_unlocked,
    softmax_probabilities,
)


class SoftmaxTests(unittest.TestCase):
    def test_known_odds_and_shift_invariance(self):
        expected = np.array([math.e, 1.0]) / (math.e + 1.0)
        np.testing.assert_allclose(softmax_probabilities([1001, 1000], temperature=1), expected)
        np.testing.assert_allclose(softmax_probabilities([1, 0], temperature=1), expected)

    def test_extreme_finite_inputs_and_no_mutation(self):
        values = np.array([np.finfo(float).max, -np.finfo(float).max])
        before = values.copy()
        np.testing.assert_array_equal(
            softmax_probabilities(values, temperature=np.nextafter(0.0, 1.0)), [1.0, 0.0]
        )
        np.testing.assert_array_equal(values, before)

    def test_invalid_input(self):
        for values in ([], [[1]], [float("nan")], [float("inf")], [1j], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                softmax_probabilities(values)
        for temperature in (0, -1, float("inf"), float("nan"), True, 10**400):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                softmax_probabilities([1], temperature=temperature)


class SupportUpdateTests(unittest.TestCase):
    def test_first_update_matches_hand_calculation(self):
        original = SupportState()
        score, new = adam_support_step(
            0.4, 0.5, original, learning_rate=0.01, beta1=0.8, beta2=0.5, epsilon=0.01
        )
        # g = -.2; m = -.04; v = .02; corrected moments are -.2 and .04.
        self.assertAlmostEqual(score, 0.4 - 0.01 * 0.2 / 0.21)
        self.assertAlmostEqual(new.m, -0.04)
        self.assertAlmostEqual(new.v, 0.02)
        self.assertEqual(new.correction_step, 1)
        self.assertEqual(original, SupportState())
        with self.assertRaises(FrozenInstanceError):
            original.m = 1

    def test_explicit_global_counter_controls_correction(self):
        state = SupportState(m=-0.02, v=0.003, correction_step=7)
        score, new = adam_support_step(0.4, 0.5, state, beta1=0.8, beta2=0.5)
        expected_m = 0.8 * -0.02 + 0.2 * -0.2
        expected_v = 0.5 * 0.003 + 0.5 * 0.04
        expected = 0.4 + 0.001 * (expected_m / (1 - 0.8**8)) / (
            math.sqrt(expected_v / (1 - 0.5**8)) + 1e-8
        )
        self.assertAlmostEqual(score, expected)
        self.assertEqual(new.correction_step, 8)
        self.assertEqual(state.correction_step, 7)

    def test_direction_neutral_case_and_clipping(self):
        for likelihood in (0.0, 0.2, 0.8):
            score, _ = adam_support_step(0.4, likelihood, SupportState())
            self.assertLess(score, 0.4)
        self.assertEqual(adam_support_step(0.4, 1, SupportState())[0], 0.4)
        self.assertEqual(adam_support_step(0.4, 0.5, SupportState(), learning_rate=10)[0], 0)

    def test_zero_decay_and_large_counter(self):
        score, new = adam_support_step(0.4, 0.5, SupportState(), beta1=0, beta2=0)
        self.assertAlmostEqual(score, 0.4 - 0.001 * 0.2 / (0.2 + 1e-8))
        self.assertEqual(new.correction_step, 1)
        self.assertTrue(math.isfinite(adam_support_step(0.4, 0.5, SupportState(correction_step=10**400))[0]))

    def test_invalid_state(self):
        for kwargs in ({"m": float("nan")}, {"v": -1}, {"v": float("inf")},
                       {"correction_step": -1}, {"correction_step": 0.5}, {"correction_step": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SupportState(**kwargs)
        with self.assertRaises(ValueError):
            adam_support_step(0.4, 0.5, None)

    def test_invalid_update_parameters(self):
        for prior, likelihood in ((-1, 0.5), (2, 0.5), (0.4, -1), (0.4, 2), (float("nan"), 0.5)):
            with self.subTest(prior=prior, likelihood=likelihood), self.assertRaises(ValueError):
                adam_support_step(prior, likelihood, SupportState())
        for kwargs in ({"learning_rate": 0}, {"beta1": 1}, {"beta2": -1},
                       {"epsilon": 0}, {"epsilon": float("inf")}, {"beta1": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adam_support_step(0.4, 0.5, SupportState(), **kwargs)
        with self.assertRaises(ValueError):
            adam_support_step(0.4, 0.5, SupportState(m=1e308), beta1=0.999999)


class PartialNormalizationTests(unittest.TestCase):
    def test_locked_mass_retained_without_global_probability_claim(self):
        scores = np.array([0.8, 0.1, 0.1])
        result = normalize_unlocked(scores, [True, False, False])
        np.testing.assert_allclose(result, [0.8, 0.5, 0.5])
        self.assertAlmostEqual(result.sum(), 1.8)
        np.testing.assert_array_equal(scores, [0.8, 0.1, 0.1])

    def test_empty_unlocked_mass_all_locked_and_all_unlocked(self):
        np.testing.assert_array_equal(normalize_unlocked([0.8, 0], [True, False]), [0.8, 0])
        np.testing.assert_array_equal(normalize_unlocked([0.8, 0.1], [True, True]), [0.8, 0.1])
        np.testing.assert_allclose(normalize_unlocked([0.2, 0.3], [False, False]), [0.4, 0.6])

    def test_invalid_masks_or_scores(self):
        for scores, mask in (([], []), ([0.2], [0]), ([0.2], [True, False]),
                             ([-0.1], [False]), ([1.1], [False]), ([float("nan")], [False])):
            with self.subTest(scores=scores, mask=mask), self.assertRaises(ValueError):
                normalize_unlocked(scores, mask)


if __name__ == "__main__":
    unittest.main()
