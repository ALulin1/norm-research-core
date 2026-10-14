"""Small, stateless descriptive-density operators, without experiment runners.

Adapted from ALulin1/norm-dynamics-reproducibility, MIT license, commit
010d8ace5d32718a64112db6cb81d4b7ba1555fc, scripts/r3_4_experiment_core.py
(kernel and Operator), and scripts/r1_3_baseline_experiment.py (SpatialOperator).
See THIRD_PARTY_NOTICES.md for the upstream copyright and license.

These are the R3 conservative no-flux/periodic operators. They are not the
original mixed-boundary experiment or a complete reproduction of any paper.
No solver, parameter feedback, clinical target, data, or evaluation is included.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def _scalar(value, name, *, positive=False, nonnegative=False):
    item = np.asarray(value, dtype=float)
    if item.ndim != 0 or not np.isfinite(item):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(item)
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _grid_and_density(x, density):
    x, density = np.asarray(x, dtype=float), np.asarray(density, dtype=float)
    if (x.ndim != 1 or len(x) < 2 or density.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(density).all()):
        raise ValueError("x and density must be matching finite 1D arrays, length >= 2")
    increments = np.diff(x)
    dx = float(increments[0])
    if dx <= 0 or not np.allclose(increments, dx, rtol=1e-10, atol=abs(dx)*1e-12):
        raise ValueError("x must be a strictly increasing uniform grid")
    return x, density, dx


def perception_kernel(offsets, mu, sigma):
    """Signed Gaussian-difference kernel at offsets ``x_j - x_i``.

    Preserve upstream conventions: zero for |mu| < 1e-10 and the signed
    derivative-Gaussian approximation for 1e-10 <= |mu| < 1e-3. For larger
    |mu|, divide the Gaussian difference by 2|mu|, then multiply by sign(mu).
    """
    offsets = np.asarray(offsets, dtype=float)
    mu, sigma = _scalar(mu, "mu"), _scalar(sigma, "sigma", positive=True)
    if not np.isfinite(offsets).all():
        raise ValueError("offsets must be finite")
    a, sign = abs(mu), np.sign(mu)
    if a < 1e-10:
        return np.zeros_like(offsets)
    if a < 1e-3:
        return sign * offsets * np.exp(-0.5*(offsets/sigma)**2) / (
            sigma**3 * np.sqrt(2*np.pi))
    return sign * (np.exp(-0.5*((offsets-a)/sigma)**2)
                   - np.exp(-0.5*((offsets+a)/sigma)**2)) / (
                       2*a*sigma*np.sqrt(2*np.pi))


def nonlocal_drift(x, density, mu, sigma, *, boundary, method="fft"):
    """Return the discrete integral G_i = integral h(y-x_i) p(y) dy.

    ``no_flux`` uses finite-domain trapezoidal quadrature, including half
    endpoint weights. ``periodic`` uses rectangle quadrature on n cells of
    period n*dx and the upstream Gaussian-image truncation (10 sigma plus
    two periods). Do not duplicate a periodic endpoint in x.

    ``direct`` and ``fft`` evaluate the same discrete operator; the finite
    FFT is zero-padded linear convolution, not circular convolution.
    Boundary labels here select the integral, not a time integration method.
    Inputs are neither clipped nor normalized and are never modified.
    """
    x, density, dx = _grid_and_density(x, density)
    mu, sigma = _scalar(mu, "mu"), _scalar(sigma, "sigma", positive=True)
    if boundary not in ("no_flux", "periodic"):
        raise ValueError("boundary must be 'no_flux' or 'periodic'")
    if method not in ("direct", "fft"):
        raise ValueError("method must be 'direct' or 'fft'")
    n = len(x)
    if boundary == "periodic":
        period = n * dx
        offsets = np.arange(n) * dx
        offsets = np.where(offsets > period/2, offsets-period, offsets)
        images = int(np.ceil((abs(mu)+10*sigma)/period)) + 2
        h = sum(perception_kernel(-offsets+m*period, mu, sigma)
                for m in range(-images, images+1))
        if method == "direct":
            indices = (np.arange(n)[:, None]-np.arange(n)[None, :]) % n
            return h[indices] @ density * dx
        return irfft(rfft(density)*rfft(h), n) * dx
    weighted = density.copy()
    weighted[[0, -1]] *= 0.5
    if method == "direct":
        offsets = (np.arange(n)[None, :]-np.arange(n)[:, None]) * dx
        return perception_kernel(offsets, mu, sigma) @ weighted * dx
    h = perception_kernel(np.arange(-(n-1), n)*dx, mu, sigma)[::-1]
    size = next_fast_len(3*n-2)
    full = irfft(rfft(weighted, size)*rfft(h, size), size)
    return full[n-1:2*n-1] * dx


def density_rhs(x, density, *, mu, sigma, diffusion, interaction, boundary,
                potential_gradient=None, method="fft"):
    """Conservative semidiscrete RHS: -div(p*(interaction*G - V') - D*p').

    Centered face fluxes use zero outer total flux for ``no_flux`` and
    wraparound for ``periodic``. The conserved discrete mass is sum(p)*dx
    for BOTH boundaries, even though no-flux G uses trapezoidal quadrature.
    ``potential_gradient`` is an explicit finite array of V'(x), or a pure
    callback taking x and returning that array. None means no potential.
    Periodic consistency of a supplied potential is the caller's responsibility.

    The centered operator alone does not guarantee positivity, stability,
    convergence, or a monotone target distance under a chosen time solver.
    No clipping, normalization, stateful feedback, or integration is applied.
    """
    x, density, dx = _grid_and_density(x, density)
    diffusion = _scalar(diffusion, "diffusion", nonnegative=True)
    interaction = _scalar(interaction, "interaction")
    velocity = interaction * nonlocal_drift(
        x, density, mu, sigma, boundary=boundary, method=method)
    if potential_gradient is not None:
        gradient = (potential_gradient(x.copy()) if callable(potential_gradient)
                    else potential_gradient)
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != x.shape or not np.isfinite(gradient).all():
            raise ValueError("potential_gradient must provide a matching finite array")
        velocity = velocity - gradient
    flux = density * velocity
    if boundary == "no_flux":
        faces = np.zeros(len(x)+1)
        faces[1:-1] = (0.5*(flux[:-1]+flux[1:])
                      - diffusion*np.diff(density)/dx)
        return -np.diff(faces)/dx
    right = 0.5*(flux+np.roll(flux, -1)) - diffusion*(np.roll(density, -1)-density)/dx
    return -(right-np.roll(right, 1))/dx
