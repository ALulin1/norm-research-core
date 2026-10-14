"""Small, data-independent control primitives for norm-density models.

Research sources: ``norm_optimization/1.2 外部势场_数据集目标训练.ipynb``,
cell 18 (tabular TD update), and ``1.3 从下到上_发表版本.ipynb``, cell 5
(PropensityMapper and prescribed surrogate construction). Cell positions are
one-based, including Markdown. These are independent, validated API
implementations of mathematical primitives, not copies of experiment runners.
No clinical coefficients, fitted parameters, training protocol or data are
included. Standard Q-learning is not claimed as a new algorithm.
"""

from __future__ import annotations

import numpy as np


def _scalar(value, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _vector(value, name: str, shape=None) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite real vector")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real vector") from exc
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite real vector")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has an incompatible shape")
    return array


def _bounds(bounds, shape):
    if len(bounds) != 2:
        raise ValueError("bounds must contain lower and upper vectors")
    lower = _vector(bounds[0], "lower bounds", shape)
    upper = _vector(bounds[1], "upper bounds", shape)
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    return lower, upper


def _integral(values, grid):
    """Trapezoidal integral, including nonuniform grid spacing."""
    return np.sum((0.5 * values[:-1] + 0.5 * values[1:]) * np.diff(grid))


def q_step(q_values, action, reward, next_q_values, *, alpha, gamma, terminal):
    """Return a copied Q row after one tabular TD update.

    The target is ``reward`` for a terminal transition and
    ``reward + gamma * max(next_q_values)`` otherwise. A terminal transition
    may use ``next_q_values=None``. All supplied rows are checked; no input is
    mutated. This function defines no exploration, state encoding or training.
    """
    values = _vector(q_values, "q_values")
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise ValueError("action must be an integer index")
    if not 0 <= action < values.size:
        raise ValueError("action is outside the Q row")
    if not isinstance(terminal, (bool, np.bool_)):
        raise ValueError("terminal must be Boolean")
    rate, discount = _scalar(alpha, "alpha"), _scalar(gamma, "gamma")
    if not 0 <= rate <= 1 or not 0 <= discount <= 1:
        raise ValueError("alpha and gamma must lie in [0, 1]")
    target = _scalar(reward, "reward")
    if next_q_values is not None:
        following = _vector(next_q_values, "next_q_values", values.shape)
    elif not terminal:
        raise ValueError("nonterminal transitions require next_q_values")
    if not terminal:
        target += discount * np.max(following)
    result = values.copy()
    result[action] += rate * (target - result[action])
    if not np.all(np.isfinite(result)):
        raise ValueError("Q update overflowed")
    return result


def density_propensity(density, grid, *, lower, upper) -> float:
    """Map a nonnegative density's centroid to an explicit propensity interval.

    For centroid position ``z`` normalized to the grid interval, return
    ``lower + (upper-lower) * sigmoid(10*(z-0.5))``. This retains the source
    PropensityMapper's map, including its asymptotic rather than exact endpoint
    saturation. Bounds must lie in [0, 1]. Trapezoidal integration supports a
    nonuniform increasing grid. Zero mass is rejected: the source notebook's
    small-mass fallback is deliberately not part of this normalized-density API.
    """
    x = _vector(grid, "grid")
    p = _vector(density, "density", x.shape)
    lo, hi = _scalar(lower, "lower"), _scalar(upper, "upper")
    if x.size < 2 or not np.all(np.diff(x) > 0) or not np.isfinite(x[-1] - x[0]):
        raise ValueError("grid must have at least two increasing finite-spaced points")
    if np.any(p < 0) or not 0 <= lo <= hi <= 1:
        raise ValueError("density must be nonnegative and bounds must lie in [0, 1]")
    mass = _integral(p, x)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("density must have finite positive integral")
    centroid = _integral((p / mass) * x, x)
    z = (centroid - x[0]) / (x[-1] - x[0])
    result = lo + (hi - lo) / (1.0 + np.exp(-10.0 * (z - 0.5)))
    if not np.isfinite(result):
        raise ValueError("density mapping overflowed")
    return float(np.clip(result, lo, hi))


def saturating_response(propensity, *, baseline, effect, rate, bounds=None):
    """Return ``baseline + effect * (1-exp(-rate*propensity))``.

    Every coefficient is supplied by the caller; optional bounds contain two
    vectors matching the baseline. This prescribed response is not an estimate
    of a causal effect. No source-study coefficients or outcome names are used.
    """
    p, k = _scalar(propensity, "propensity"), _scalar(rate, "rate")
    if not 0 <= p <= 1 or k <= 0:
        raise ValueError("propensity must lie in [0, 1] and rate must be positive")
    base = _vector(baseline, "baseline")
    effects = _vector(effect, "effect", base.shape)
    result = base + effects * (-np.expm1(-k * p))
    if not np.all(np.isfinite(result)):
        raise ValueError("response overflowed")
    if bounds is not None:
        result = np.clip(result, *_bounds(bounds, base.shape))
    return result


def surrogate_score(values, baseline, *, direction, scale, gain_weight,
                    loss_weight, gain_cap, bonus=0.0, penalty=0.0,
                    clip=None, control_cost=0.0) -> float:
    """Score prescribed changes with caller-supplied directions and weights.

    Let ``z = direction * (values-baseline) / scale``, with direction +1 for
    increases and -1 for decreases. Positive terms are ``gain_weight *
    min(z, gain_cap)``; nonpositive terms are ``-loss_weight * abs(z)``.
    Add the explicit bonus and subtract the explicit penalty, optionally clip
    that subtotal, then subtract control_cost. No bonus/penalty triggers or
    experimental thresholds are inferred. As in the source construction,
    subtracting cost after clipping can put the final score below the clip.
    """
    value = _vector(values, "values")
    base = _vector(baseline, "baseline", value.shape)
    sign = _vector(direction, "direction", value.shape)
    unit = _vector(scale, "scale", value.shape)
    gain = _vector(gain_weight, "gain_weight", value.shape)
    loss = _vector(loss_weight, "loss_weight", value.shape)
    cap = _vector(gain_cap, "gain_cap", value.shape)
    if np.any(np.abs(sign) != 1) or np.any(unit <= 0):
        raise ValueError("direction must be +/-1 and scale must be positive")
    if np.any(gain < 0) or np.any(loss < 0) or np.any(cap < 0):
        raise ValueError("weights and gain caps must be nonnegative")
    extra, deduction = _scalar(bonus, "bonus"), _scalar(penalty, "penalty")
    cost = _scalar(control_cost, "control_cost")
    if extra < 0 or deduction < 0 or cost < 0:
        raise ValueError("bonus, penalty and control_cost must be nonnegative")
    z = sign * (value - base) / unit
    terms = np.where(z > 0, gain * np.minimum(z, cap), -loss * np.abs(z))
    subtotal = np.sum(terms) + extra - deduction
    if not np.isfinite(subtotal):
        raise ValueError("surrogate subtotal overflowed")
    if clip is not None:
        if len(clip) != 2:
            raise ValueError("clip must contain two scalar bounds")
        lo, hi = _scalar(clip[0], "clip lower"), _scalar(clip[1], "clip upper")
        if lo > hi:
            raise ValueError("clip lower must not exceed clip upper")
        subtotal = np.clip(subtotal, lo, hi)
    result = float(subtotal - cost)
    if not np.isfinite(result):
        raise ValueError("surrogate score overflowed")
    return result
