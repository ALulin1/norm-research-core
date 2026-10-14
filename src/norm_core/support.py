"""Small support-score kernels associated with the JBI norm-induction study.

Academic source: C. Li, O. Petruchik, E. Grishanina and S. Kovalchuk,
``Multi-agent norm perception and induction in distributed healthcare`` (2025),
doi:10.1016/j.jbi.2025.104835.

The retained implementation uses action softmax probabilities and an Adam-like
support update (notebook cell 5, lines 288--324), then normalizes only unlocked
scores (lines 334--344). These independently organized, array/scalar APIs expose
those arithmetic rules without clinical labels, agents, experiments or data.
They are not Bayesian posterior inference or the complete locking experiment.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(values, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0 or result.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real numeric vector")
    result = result.astype(float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def softmax_probabilities(action_values, *, temperature: float = 5.0) -> np.ndarray:
    """Return stable action probabilities; leave the input unchanged.

    Temperature 5.0 is the retained environment default. Maximum subtraction is
    an arithmetic stabilization; no action selection or likelihood aggregation
    is performed here. Very small probabilities may round to zero.
    """
    values = _vector(action_values, "action_values")
    temperature = _finite_scalar(temperature, "temperature")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with np.errstate(over="ignore", under="ignore"):
        exponentials = np.exp((values - values.max()) / temperature)
    return exponentials / exponentials.sum()


@dataclass(frozen=True)
class SupportState:
    """Scalar moments and the correction counter preceding the next update.

    The retained notebook uses an agent-global counter across selected norms.
    Callers preserving that convention must supply the current global count
    together with the selected norm's moments. This object does not implement
    the experiment's scheduling or maintain other norms' states.
    """

    m: float = 0.0
    v: float = 0.0
    correction_step: int = 0

    def __post_init__(self) -> None:
        m = _finite_scalar(self.m, "m")
        v = _finite_scalar(self.v, "v")
        if v < 0:
            raise ValueError("v must be nonnegative")
        step = self.correction_step
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0:
            raise ValueError("correction_step must be a nonnegative integer")
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "correction_step", int(step))


def _bias_denominator(beta: float, step: int) -> float:
    if beta == 0:
        return 1.0
    try:
        exponent = step * math.log(beta)
    except OverflowError:
        return 1.0  # For beta < 1, beta**step tends to zero.
    return -math.expm1(exponent)


def adam_support_step(
    prior: float,
    likelihood: float,
    state: SupportState,
    *,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[float, SupportState]:
    """Return a clipped support score and a new immutable moment state.

    The retained rule is g = prior * (likelihood - 1), followed by moment
    correction and *addition* of the adaptive increment. Here g is a heuristic
    discrepancy, not an asserted objective gradient. For probability-valued
    inputs, g is nonpositive; prior momentum also affects the update direction.
    This function performs neither locking nor cross-score normalization.
    Defaults match the retained active update. Bias correction has its usual
    zero-initialized-moment interpretation; supplied states are explicit.
    """
    prior = _finite_scalar(prior, "prior")
    likelihood = _finite_scalar(likelihood, "likelihood")
    learning_rate = _finite_scalar(learning_rate, "learning_rate")
    beta1 = _finite_scalar(beta1, "beta1")
    beta2 = _finite_scalar(beta2, "beta2")
    epsilon = _finite_scalar(epsilon, "epsilon")
    if not 0 <= prior <= 1 or not 0 <= likelihood <= 1:
        raise ValueError("prior and likelihood must lie in [0, 1]")
    if learning_rate <= 0 or epsilon <= 0:
        raise ValueError("learning_rate and epsilon must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    if not isinstance(state, SupportState):
        raise ValueError("state must be a SupportState")

    step = state.correction_step + 1
    discrepancy = prior * (likelihood - 1.0)
    m = beta1 * state.m + (1.0 - beta1) * discrepancy
    v = beta2 * state.v + (1.0 - beta2) * discrepancy**2
    m_hat = m / _bias_denominator(beta1, step)
    v_hat = v / _bias_denominator(beta2, step)
    if not math.isfinite(m_hat) or not math.isfinite(v_hat):
        raise ValueError("corrected moments exceed finite floating-point range")
    updated = prior + learning_rate * (m_hat / (math.sqrt(v_hat) + epsilon))
    if not math.isfinite(updated):
        raise ValueError("support update exceeds finite floating-point range")
    return min(max(updated, 0.0), 1.0), SupportState(m, v, step)


def normalize_unlocked(scores, locked) -> np.ndarray:
    """Normalize only unlocked scores, retaining locked values exactly.

    ``locked`` is an explicit Boolean mask. Positive unlocked mass is rescaled
    to one, even when locked scores already carry mass. Thus the whole vector
    generally does NOT sum to one. Zero unlocked mass is left unchanged.
    The caller supplies already stored locked values; threshold detection and
    the original experiment's lock-state updates are deliberately absent.
    """
    result = _vector(scores, "scores")
    if np.any(result < 0) or np.any(result > 1):
        raise ValueError("scores must lie in [0, 1]")
    mask = np.asarray(locked)
    if mask.dtype.kind != "b" or mask.shape != result.shape:
        raise ValueError("locked must be a Boolean mask matching scores")
    unlocked = ~mask
    mass = result[unlocked].sum()
    if mass > 0:
        result[unlocked] /= mass
    return result
