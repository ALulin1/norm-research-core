"""Tests of control primitives using tiny constructed arrays only."""

import unittest

import numpy as np

from norm_core.control import (
    density_propensity, q_step, saturating_response, surrogate_score,
)


class ControlTests(unittest.TestCase):
    def test_terminal_target_has_no_bootstrap_and_keeps_input(self):
        before = np.array([2.0, 4.0])
        after = q_step(before, 0, -2, [100, 200], alpha=0.25,
                       gamma=0.9, terminal=True)
        np.testing.assert_allclose(after, [1, 4])
        np.testing.assert_array_equal(before, [2, 4])
        np.testing.assert_allclose(
            q_step(before, 0, -2, None, alpha=0.25, gamma=0.9, terminal=True),
            after,
        )

    def test_nonterminal_target_and_update_direction(self):
        result = q_step([2, 4], 0, 1, [3, 6], alpha=0.5,
                        gamma=0.5, terminal=False)
        np.testing.assert_allclose(result, [3, 4])
        result = q_step(result, 0, -3, None, alpha=1,
                        gamma=0, terminal=True)
        np.testing.assert_allclose(result, [-3, 4])

    def test_q_invalid_inputs(self):
        base = dict(q_values=[0, 0], action=0, reward=1, next_q_values=[1, 2],
                    alpha=0.5, gamma=0.9, terminal=False)
        for change in [dict(action=-1), dict(action=True), dict(alpha=1.1),
                       dict(gamma=-0.1), dict(terminal=1), dict(reward=np.inf),
                       dict(q_values=[[0, 0]]), dict(next_q_values=[0]),
                       dict(next_q_values=None), dict(q_values=[0, np.nan])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                q_step(**(base | change))

    def test_propensity_normalizes_mass_and_nonuniform_grid(self):
        x = np.array([0.0, 0.25, 1.0])
        p = np.ones(3)
        self.assertAlmostEqual(density_propensity(p, x, lower=0.2, upper=0.8), 0.5)
        self.assertAlmostEqual(
            density_propensity(p * 1e-20, x, lower=0.2, upper=0.8), 0.5,
        )

    def test_propensity_endpoints_are_sigmoid_not_hard_thresholds(self):
        low = density_propensity([1, 0], [0, 1], lower=0.2, upper=0.8)
        high = density_propensity([0, 1], [0, 1], lower=0.2, upper=0.8)
        self.assertAlmostEqual(low, 0.2 + 0.6 / (1 + np.exp(5)))
        self.assertAlmostEqual(high, 0.2 + 0.6 / (1 + np.exp(-5)))
        self.assertGreater(low, 0.2)
        self.assertLess(high, 0.8)
        self.assertAlmostEqual(
            density_propensity([1, 1], [0, 1], lower=0.3, upper=0.3), 0.3,
        )

    def test_propensity_is_affine_grid_invariant(self):
        p, x = [0, 1, 3], np.array([0, 0.5, 1])
        self.assertAlmostEqual(
            density_propensity(p, x, lower=0.1, upper=0.7),
            density_propensity(p, 4 * x + 2, lower=0.1, upper=0.7),
        )

    def test_propensity_invalid_density_and_grid(self):
        cases = [([0, 0], [0, 1]), ([-1, 2], [0, 1]), ([1], [0]),
                 ([1, 1], [0, 0]), ([1, 1], [1, 0]), ([1, 1], [0, np.nan]),
                 ([1, 1, 1], [0, 1]), ([1, np.inf], [0, 1])]
        for p, x in cases:
            with self.subTest(p=p, x=x), self.assertRaises(ValueError):
                density_propensity(p, x, lower=0, upper=1)
        for lo, hi in [(-1, 1), (0, 2), (0.8, 0.2), (np.nan, 1)]:
            with self.subTest(lower=lo, upper=hi), self.assertRaises(ValueError):
                density_propensity([1, 1], [0, 1], lower=lo, upper=hi)

    def test_response_baseline_direction_and_explicit_bounds(self):
        args = dict(baseline=[2, 3], effect=[1, -1], rate=2)
        np.testing.assert_allclose(saturating_response(0, **args), [2, 3])
        value = saturating_response(0.5, **args)
        self.assertGreater(value[0], 2)
        self.assertLess(value[1], 3)
        np.testing.assert_allclose(
            saturating_response(0.5, **args, bounds=([0, 2.8], [2.2, 4])),
            [2.2, 2.8],
        )

    def test_response_invalid_inputs(self):
        args = dict(propensity=0.5, baseline=[1, 2], effect=[1, -1], rate=2)
        for change in [dict(propensity=-0.1), dict(propensity=1.1), dict(rate=0),
                       dict(effect=[1]), dict(effect=[1, np.nan]),
                       dict(bounds=([2, 0], [1, 2])),
                       dict(bounds=([0], [1]))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                saturating_response(**(args | change))

    def test_score_directions_and_asymmetric_weights(self):
        value = surrogate_score(
            [3, 1, 4], [1, 2, 2], direction=[1, -1, -1], scale=[1, 1, 2],
            gain_weight=[2, 3, 4], loss_weight=[5, 6, 7], gain_cap=[1, 2, 3],
        )
        self.assertAlmostEqual(value, -2)  # 2 + 3 - 7

    def test_score_clips_before_control_cost(self):
        args = dict(direction=[1], scale=[1], gain_weight=[2],
                    loss_weight=[3], gain_cap=[10])
        self.assertAlmostEqual(surrogate_score(
            [0], [1], **args, bonus=1, penalty=2, clip=(-1, 1), control_cost=0.5,
        ), -1.5)

    def test_score_invalid_inputs(self):
        args = dict(values=[2], baseline=[1], direction=[1], scale=[1],
                    gain_weight=[2], loss_weight=[3], gain_cap=[4])
        for change in [dict(scale=[0]), dict(direction=[0]), dict(gain_weight=[-1]),
                       dict(loss_weight=[-1]), dict(gain_cap=[-1]), dict(bonus=-1),
                       dict(penalty=-1), dict(control_cost=-1), dict(clip=(2, 1)),
                       dict(values=[np.nan]), dict(baseline=[1, 2])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                surrogate_score(**(args | change))


if __name__ == "__main__":
    unittest.main()
