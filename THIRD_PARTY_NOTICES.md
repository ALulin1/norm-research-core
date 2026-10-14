# Sources and attribution

## Numerical and perception kernels

Adapted from **Chao Li and contributors**, *Norm dynamics: selected numerical
algorithms and supplementary experiments*, MIT license, fixed commit
`010d8ace5d32718a64112db6cb81d4b7ba1555fc`:

- [`scripts/r3_4_experiment_core.py`](https://github.com/ALulin1/norm-dynamics-reproducibility/blob/010d8ace5d32718a64112db6cb81d4b7ba1555fc/scripts/r3_4_experiment_core.py), `kernel` and `Operator`.
- [`scripts/r1_3_baseline_experiment.py`](https://github.com/ALulin1/norm-dynamics-reproducibility/blob/010d8ace5d32718a64112db6cb81d4b7ba1555fc/scripts/r1_3_baseline_experiment.py), `perception_kernel`, `SpatialOperator`, `kde_values`, `fit_sinp`, `sample_gmm` and distance definitions in `measure`.

Copyright (c) 2026 Chao Li and contributors. The complete MIT permission and
warranty notice is retained in [LICENSE](LICENSE), with the same attribution.
The upstream archive is [doi:10.5281/zenodo.22804562](https://doi.org/10.5281/zenodo.22804562).
Its data terms are separate; none of its input or result files is included here.

Changes include scalar/single-window APIs, explicit boundary choices, input
validation, caller-owned random state and removal of I/O and experiment policy.
Direct and FFT paths share the same discrete operator. Local mixture settings
are explicit; random streams do not claim historical execution identity.

## JBI support arithmetic

Academic source: **Chao Li, Olga Petruchik, Elizaveta Grishanina and Sergey
Kovalchuk**, *Multi-agent norm perception and induction in distributed healthcare*
(2025), [doi:10.1016/j.jbi.2025.104835](https://doi.org/10.1016/j.jbi.2025.104835).

The support module independently organizes the verified action-softmax,
Adam-like arithmetic and partial-normalization rules as generic scalar/array
functions. Source audit: retained norm-induction notebook, cell 5, lines
288–324 and 334–344; file SHA-256
`fd891967d29b6b85fe14907f3bda3c7a3387e503251e46ac7d6a3492313cdcd0`.
Clinical rules, agents and the medical application repository are not copied.
This package's license does not relicense those repositories or clinical material.

## Q-control arithmetic

Generic interfaces follow the retained target-matching notebook cell 18 and
proxy-reward notebook cell 5. Audited source SHA-256 values:

- Target matching: `fa24fe7ed799d5687ffd9eb71b391f730b9c36441d56b437ab1f986b7fab9801`.
- Proxy reward: `f459fca7055670408bfca93cfdb538467575e07fcb9fc8beb9821e67329f7fd8`.

The single TD step is standard tabular Q-learning. The density mapping,
saturating response and surrogate aggregation are parameterized primitives.
Experimental policies and clinical coefficients are supplied by neither this
release nor its tests. No new scientific priority is claimed for standard tools.

NumPy and SciPy are external dependencies and retain their own licenses.
