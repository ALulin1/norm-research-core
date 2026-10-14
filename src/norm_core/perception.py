"""Local KDE/peak mixture reconstruction and separately named distance helpers.

Adapted from the MIT numerical package at commit
010d8ace5d32718a64112db6cb81d4b7ba1555fc: r1_3_baseline_experiment.py,
kde_values, fit_sinp, sample_gmm and measure. See THIRD_PARTY_NOTICES.md.
This is a small single-window API, without an experiment or feedback schedule.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance


def _vector(a, name):
    a = np.asarray(a)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real numeric vector")
    a = a.astype(float, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return a


def _count(n, name, minimum=1):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(n)


@dataclass(frozen=True)
class Mixture1D:
    """An explicit caller-owned Gaussian mixture; no fitted parameters included."""
    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def __post_init__(self):
        arrays = [_vector(v, n).copy() for v, n in
                  [(self.weights, "weights"), (self.means, "means"), (self.stds, "stds")]]
        w, m, s = arrays
        if w.shape != m.shape or w.shape != s.shape:
            raise ValueError("mixture vectors must have matching shapes")
        if np.any(w < 0) or not np.isclose(w.sum(), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("weights must be nonnegative and sum to one")
        if np.any(s <= 0):
            raise ValueError("stds must be positive")
        for name, value in zip(("weights", "means", "stds"), arrays):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def pdf(self, points):
        if np.iscomplexobj(points):
            raise ValueError("points must be real")
        points = np.asarray(points, dtype=float)
        if not np.isfinite(points).all():
            raise ValueError("points must be finite")
        z = (points[..., None] - self.means) / self.stds
        return (self.weights * np.exp(-0.5*z*z) /
                (self.stds*np.sqrt(2*np.pi))).sum(axis=-1)

    def sample(self, size, rng):
        """Draw from an explicitly supplied NumPy Generator."""
        size = _count(size, "size")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a NumPy Generator")
        components = np.searchsorted(np.cumsum(self.weights), rng.random(size), side="right")
        components = np.minimum(components, len(self.weights)-1)
        return self.means[components] + self.stds[components]*rng.standard_normal(size)


def local_mixture(x, density, *, center_index, radius, sample_count, kde_points,
                  peak_min_height, peak_distance, variance_floor, rng):
    """Reconstruct one local mixture with the retained KDE/peak heuristic.

    All window and fit settings are explicit. Nonnegative density weights select
    grid points; a zero-mass window uses uniform sampling. The sample count is
    capped by the number of window points, as in the retained implementation.
    Scott KDE uses ddof=1. A degenerate sample falls back to one component.

    The retained width convention is std = max(half-height width / 2,
    sqrt(variance_floor)), NOT the Gaussian FWHM-to-sigma conversion. No EM fit,
    likelihood optimum, original random stream or paper-result reproduction is
    asserted. Returns (Mixture1D, status), with an explicit fallback status.
    """
    x, p = _vector(x, "x"), _vector(density, "density")
    if x.shape != p.shape or len(x) < 2 or np.any(p < 0):
        raise ValueError("x and nonnegative density must match, length >= 2")
    dx = np.diff(x)
    if dx[0] <= 0 or not np.allclose(dx, dx[0], rtol=1e-10, atol=abs(dx[0])*1e-12):
        raise ValueError("x must be a strictly increasing uniform grid")
    center_index = _count(center_index, "center_index", 0)
    if center_index >= len(x):
        raise ValueError("center_index exceeds the grid")
    sample_count, kde_points = _count(sample_count, "sample_count"), _count(kde_points, "kde_points", 3)
    peak_distance = _count(peak_distance, "peak_distance")
    for value, name in [(radius, "radius"), (peak_min_height, "peak_min_height"),
                        (variance_floor, "variance_floor")]:
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar")
    if variance_floor == 0 or not isinstance(rng, np.random.Generator):
        raise ValueError("variance_floor must be positive and rng a NumPy Generator")
    steps = len(x) if radius >= dx[0]*(len(x)-1) else int(radius/dx[0])
    lo, hi = max(0, center_index-steps), min(len(x), center_index+steps+1)
    count = min(sample_count, hi-lo)
    weights = p[lo:hi]
    if weights.max() > 0:
        weights = weights/weights.max()  # Preserve probabilities without overflow.
        cumulative = np.r_[0.0, np.cumsum(weights)]
        indices = np.searchsorted(cumulative, rng.random(count)*cumulative[-1], side="right")-1
    else:
        indices = (rng.random(count)*(hi-lo)).astype(int)
    samples = x[lo+np.clip(indices, 0, hi-lo-1)]
    floor = np.sqrt(variance_floor)
    if count < 2 or np.ptp(samples) <= 1e-12 or samples.std(ddof=1) <= 1e-12:
        return Mixture1D([1.0], [samples[0]], [floor]), "degenerate"
    grid = np.linspace(samples.min(), samples.max(), kde_points)
    bandwidth = max(samples.std(ddof=1)*count**(-0.2), 1e-12)
    z = (grid[:, None]-samples[None, :])/bandwidth
    curve = np.exp(-0.5*z*z).mean(axis=1)/(bandwidth*np.sqrt(2*np.pi))
    peaks, properties = find_peaks(curve, height=peak_min_height, distance=peak_distance)
    if len(peaks) == 0:
        return Mixture1D([1.0], [grid[np.argmax(curve)]], [floor]), "no_peaks"
    stds = []
    for peak in peaks:
        left = right = peak
        half = curve[peak]/2
        while left > 0 and curve[left] > half:
            left -= 1
        while right < kde_points-1 and curve[right] > half:
            right += 1
        stds.append(np.sqrt(max((grid[right]-grid[left])**2/4, variance_floor)))
    heights = properties["peak_heights"]
    return Mixture1D(heights/heights.sum(), grid[peaks], stds), "peaks"


def sample_wasserstein(a, b):
    """1D W1 of empirical samples; this is not a density-weighted local mean."""
    return float(wasserstein_distance(_vector(a, "a"), _vector(b, "b")))


def global_grid_distance(x, weights, target_samples):
    """W1 of weighted grid POINT MASSES versus target samples (no quadrature)."""
    x, weights = _vector(x, "x"), _vector(weights, "weights")
    if x.shape != weights.shape or np.any(weights < 0) or weights.max() <= 0:
        raise ValueError("weights must match x and have positive nonnegative mass")
    weights = weights/weights.max()
    return float(wasserstein_distance(x, _vector(target_samples, "target_samples"), u_weights=weights))


def local_distance_mean(distances, population_weights):
    """Population-weighted mean of local discrepancies; distinct from global W1."""
    a, w = _vector(distances, "distances"), _vector(population_weights, "population_weights")
    if a.shape != w.shape or np.any(a < 0) or np.any(w < 0) or w.max() <= 0:
        raise ValueError("matching nonnegative vectors and positive total weight required")
    w = w/w.max()
    return float(np.dot(a, w/w.sum()))
