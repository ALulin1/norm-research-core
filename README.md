# Norm Research Core

A compact Python library for norm perception, density dynamics, parameter-control
primitives and prescriptive support updates. It brings selected research kernels
behind explicit NumPy array interfaces. Version **0.1.0**.

## Components

| Module | Core operations | Research role |
|---|---|---|
| `density` | Signed perception kernel, matched direct/FFT nonlocal integral, conservative no-flux/periodic RHS | Descriptive norm dynamics; Claim 1 foundation |
| `perception` | Local sampling → Scott KDE → peak-based Gaussian mixture; local and global distance helpers | Perception reconstruction and discrepancy; Claim 1 foundation |
| `control` | One Q update, density-to-propensity map, explicit saturating response and surrogate score | Primitives used by target matching and proxy-reward control; Claims 1–2 |
| `support` | Stable action softmax, explicit moment/counter state, Adam-like support update and partial normalization | Prescriptive support; Claim 3 foundation |

The integration contribution is a small, installable API with input checks and
synthetic component tests. Standard numerical and learning operations remain
standard methods. This release does not add a fourth scientific claim.

## Install and use

Python 3.10+, NumPy 1.26+ and SciPy 1.14+ are required.

```sh
python -m pip install "git+https://github.com/ALulin1/norm-research-core.git@v0.1.0"
```

All values below are artificial demonstration inputs:

```python
import numpy as np
from norm_core import density_rhs, SupportState, adam_support_step

x = np.linspace(-2.0, 2.0, 17)
p = np.exp(-x*x)
rhs = density_rhs(x, p, mu=0.4, sigma=0.3, diffusion=0.1,
                  interaction=0.2, boundary="no_flux")
score, state = adam_support_step(0.5, 0.8, SupportState())
```

Functions consume arrays/scalars and return arrays/scalars or explicit state.
Random sampling requires a caller-provided `numpy.random.Generator`. There are
no implicit file reads, network calls, experiment runners or model downloads.
Function docstrings specify shapes, parameters and numerical conventions.

## Scope and conventions

- The numerical RHS follows the conservative R3 controls. The historical mixed
  boundary solver and stateful feedback inside solver calls are separate implementations.
  No time solver, positivity/convergence guarantee or trained controller is supplied.
- Local reconstruction preserves the retained half-height-width convention
  (`std = width / 2`, with an explicit variance floor). It is a peak heuristic,
  not an EM fit or the Gaussian FWHM-to-sigma conversion. A single-window API
  changes random-number consumption relative to the original batch experiments.
- Local mean discrepancy, empirical sample W1 and global weighted-grid W1 have
  separate names. Grid weights are point masses; nonuniform-grid quadrature is
  not inferred automatically.
- Control mappings use caller-specified bounds, coefficients and weights.
  Clinical coefficients, treatments, reward-trigger rules, training schedules,
  replay and evaluation protocols are absent. The strict propensity API rejects
  zero-mass densities instead of returning the original experiment's fallback.
- Support scores are heuristic quantities. Bias correction takes an explicit
  counter; reproducing agent-global timing requires the caller to manage it.
  Only unlocked scores are normalized, so the full vector need not sum to one.

No patient records, aggregate empirical inputs, fitted targets, checkpoints,
experiment configurations, notebook outputs, reference results or figures are
distributed. Tests construct small artificial arrays in memory. The separate
DyMES, Heatbugs, INRC and GitHub-analysis workflows remain outside this core.
These component tests do not reproduce the papers' experiments or validate
clinical effectiveness.

## Develop and test

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests check signed kernels, direct/FFT agreement, discrete mass conservation,
local reconstruction edge cases, distinct distance definitions, explicit Q
termination, response bounds, support-update direction and partial normalization.

## Sources

Numerical kernels were adapted from the MIT-licensed
[norm-dynamics-reproducibility](https://github.com/ALulin1/norm-dynamics-reproducibility/tree/010d8ace5d32718a64112db6cb81d4b7ba1555fc)
release. JBI support and retained Q-control arithmetic were reorganized as generic
interfaces. See [source and attribution notes](THIRD_PARTY_NOTICES.md) for the
specific provenance and adaptation boundaries. License: [MIT](LICENSE).
