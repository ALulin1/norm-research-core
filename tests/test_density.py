"""Synthetic operator checks; no archived data, experiment, or solver is run."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from norm_core.density import density_rhs, nonlocal_drift, perception_kernel


class DensityTests(unittest.TestCase):
    def test_signed_kernel_oddness_and_sign_reversal(self):
        y = np.linspace(-2, 2, 51)
        for mu in [0.0, 1e-12, 2e-4, 0.7]:
            with self.subTest(mu=mu):
                positive = perception_kernel(y, mu, 0.3)
                assert_allclose(perception_kernel(y, -mu, 0.3), -positive, atol=1e-15)
                assert_allclose(perception_kernel(-y, mu, 0.3), -positive, atol=1e-15)

    def test_direct_fft_and_conservative_mass(self):
        x = np.arange(65)*0.05
        shapes = [np.ones_like(x), np.exp(-((x-1.1)/0.08)**2),
                  0.1+x/10+np.exp(-((x-0.25)/0.16)**2)]
        for boundary in ["no_flux", "periodic"]:
            for p in shapes:
                with self.subTest(boundary=boundary, peak=float(p.max())):
                    original = p.copy()
                    direct = nonlocal_drift(x, p, -0.4, 0.12, boundary=boundary, method="direct")
                    fft = nonlocal_drift(x, p, -0.4, 0.12, boundary=boundary, method="fft")
                    assert_allclose(fft, direct, rtol=2e-13, atol=2e-14)
                    rhs = density_rhs(x, p, mu=-0.4, sigma=0.12, diffusion=0.03,
                                      interaction=0.8, boundary=boundary,
                                      potential_gradient=0.2*np.sin(2*np.pi*x/(len(x)*0.05)))
                    self.assertLess(abs(rhs.sum()*0.05), 2e-13)
                    assert_allclose(p, original, rtol=0, atol=0)

    def test_periodic_wraparound_and_constant_equilibrium(self):
        x = np.arange(64)*0.05
        impulse = np.zeros_like(x)
        impulse[0] = 1.0
        g = nonlocal_drift(x, impulse, 0.2, 0.1, boundary="periodic")
        shifted = nonlocal_drift(x, np.roll(impulse, 5), 0.2, 0.1, boundary="periodic")
        assert_allclose(shifted, np.roll(g, 5), atol=1e-15)
        assert_allclose(g[-1], -g[1], atol=1e-15)
        rhs = density_rhs(x, np.ones_like(x), mu=0.2, sigma=0.1,
                          diffusion=0.1, interaction=1.0, boundary="periodic")
        assert_allclose(rhs, 0, atol=1e-14)

    def test_finite_integral_orientation_and_endpoint_weights(self):
        x = np.array([0.0, 0.25, 0.5])
        p = np.array([1.0, 2.0, 4.0])
        expected = np.array([sum(perception_kernel(xj-xi, 0.3, 0.1)*pj*w
                                 for xj, pj, w in zip(x, p, [0.5, 1, 0.5]))*0.25
                             for xi in x])
        assert_allclose(nonlocal_drift(x, p, 0.3, 0.1, boundary="no_flux"), expected)

    def test_no_flux_potential_direction_and_callback(self):
        x, p = np.arange(5.0), np.ones(5)
        options = dict(mu=0.0, sigma=1.0, diffusion=0.0, interaction=0.0,
                       boundary="no_flux")
        rhs = density_rhs(x, p, **options, potential_gradient=lambda grid: 2*(grid-2))
        assert_allclose(rhs, [-3, 2, 2, 2, -3])
        assert_allclose(rhs, density_rhs(x, p, **options, potential_gradient=2*(x-2)))

    def test_invalid_grid_and_density_rejected(self):
        cases = [([0, 1], [1]), ([0, 1], [[1, 2]]),
                 ([0, np.nan], [1, 2]), ([0, 1], [1, np.inf]),
                 ([0, 1, 3], [1, 2, 3]), ([1, 0], [1, 2]), ([0], [1])]
        for x, p in cases:
            with self.subTest(x=x, p=p), self.assertRaises(ValueError):
                nonlocal_drift(x, p, 0.2, 0.1, boundary="no_flux")

    def test_invalid_operator_parameters_rejected(self):
        cases = [{"sigma": 0}, {"mu": np.inf}, {"diffusion": -1},
                 {"interaction": np.nan}, {"boundary": "mixed"}, {"method": "unknown"},
                 {"potential_gradient": [1, 2]}, {"potential_gradient": [1, 2, np.nan]}]
        for kwargs in cases:
            options = dict(mu=0.2, sigma=0.1, diffusion=0.1, interaction=1,
                           boundary="no_flux")
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                density_rhs([0, 1, 2], [1, 2, 1], **options)


if __name__ == "__main__":
    unittest.main()
