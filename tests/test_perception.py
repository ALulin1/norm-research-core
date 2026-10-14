import unittest
import numpy as np
from norm_core.perception import (Mixture1D, local_mixture, sample_wasserstein,
                                  global_grid_distance, local_distance_mean)


class PerceptionTests(unittest.TestCase):
    def test_mixture_is_a_normalized_density(self):
        model = Mixture1D([0.25, 0.75], [-1, 2], [0.5, 1])
        x = np.linspace(-10, 12, 4001)
        p = model.pdf(x)
        mass = np.sum((p[:-1]+p[1:])*np.diff(x)/2)
        self.assertAlmostEqual(mass, 1, places=10)
        self.assertFalse(model.weights.flags.writeable)

    def test_explicit_random_state(self):
        model = Mixture1D([1], [0], [1])
        np.testing.assert_array_equal(model.sample(8, np.random.default_rng(3)),
                                      model.sample(8, np.random.default_rng(3)))

    def fit(self, p, **kw):
        opts = dict(center_index=2, radius=2, sample_count=5, kde_points=31,
                    peak_min_height=0, peak_distance=2, variance_floor=0.01,
                    rng=np.random.default_rng(7))
        opts.update(kw)
        return local_mixture(np.arange(5.0), p, **opts)

    def test_degenerate_window(self):
        model, status = self.fit([0, 0, 1, 0, 0])
        self.assertEqual(status, "degenerate")
        np.testing.assert_array_equal(model.means, [2])
        np.testing.assert_allclose(model.stds, [0.1])

    def test_empty_window_is_finite_and_explicit(self):
        model, status = self.fit([0]*5)
        self.assertTrue(np.isfinite(model.pdf([0, 1])).all())
        self.assertIn(status, ("peaks", "no_peaks", "degenerate"))

    def test_no_peak_fallback(self):
        model, status = self.fit(np.ones(5), peak_min_height=1e6)
        self.assertEqual(status, "no_peaks")
        np.testing.assert_allclose(model.stds, [0.1])

    def test_distances_have_distinct_meanings(self):
        self.assertAlmostEqual(sample_wasserstein([0, 1], [2, 3]), 2)
        self.assertAlmostEqual(global_grid_distance([0, 2], [1, 1], [1]), 1)
        self.assertAlmostEqual(local_distance_mean([1, 3], [3, 1]), 1.5)

    def test_inputs_are_not_mutated(self):
        p = np.arange(1.0, 6)
        before = p.copy()
        self.fit(p)
        np.testing.assert_array_equal(p, before)

    def test_weight_scale_does_not_overflow(self):
        self.assertAlmostEqual(local_distance_mean([1, 3], [1e308, 1e308]), 2)
        self.assertAlmostEqual(global_grid_distance([0, 2], [1e308, 1e308], [1]), 1)
        first, a = self.fit(np.ones(5))
        second, b = self.fit(np.full(5, 1e308))
        self.assertEqual(a, b)
        np.testing.assert_allclose(first.weights, second.weights)
        np.testing.assert_allclose(first.means, second.means)
        np.testing.assert_allclose(first.stds, second.stds)

    def test_complex_values_are_rejected(self):
        with self.assertRaises(ValueError): Mixture1D(np.array([1+1j]), [0], [1])
        with self.assertRaises(ValueError): sample_wasserstein([1+1j], [0])
        with self.assertRaises(ValueError): Mixture1D([1], [0], [1]).pdf([1+1j])

    def test_invalid_inputs(self):
        for args in [([1], [0], [0]), ([2], [0], [1]), ([1], [np.nan], [1])]:
            with self.assertRaises(ValueError): Mixture1D(*args)
        for kw in [dict(radius=-1), dict(variance_floor=0), dict(center_index=5),
                   dict(kde_points=2), dict(rng=None), dict(sample_count=True)]:
            with self.assertRaises(ValueError): self.fit(np.ones(5), **kw)
        with self.assertRaises(ValueError): self.fit([1, -1, 1, 1, 1])
        with self.assertRaises(ValueError): local_distance_mean([1], [0])
        with self.assertRaises(ValueError): global_grid_distance([0], [0], [0])


if __name__ == "__main__":
    unittest.main()
