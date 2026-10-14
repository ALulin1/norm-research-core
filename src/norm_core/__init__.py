"""Norm Research Core: small research primitives with explicit array interfaces."""
from .density import density_rhs, nonlocal_drift, perception_kernel
from .perception import (Mixture1D, local_mixture, sample_wasserstein,
                         global_grid_distance, local_distance_mean)
from .control import q_step, density_propensity, saturating_response, surrogate_score
from .support import SupportState, softmax_probabilities, adam_support_step, normalize_unlocked

__version__ = "0.1.0"
__all__ = [
    "density_rhs", "nonlocal_drift", "perception_kernel", "Mixture1D", "local_mixture",
    "sample_wasserstein", "global_grid_distance", "local_distance_mean", "q_step",
    "density_propensity", "saturating_response", "surrogate_score", "SupportState",
    "softmax_probabilities", "adam_support_step", "normalize_unlocked",
]
