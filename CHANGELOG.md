# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2] - 2026-09-05

### Added

- `inference.ols`: log-linear growth rate with a t-based interval and doubling time (rung 0).
- `inference.nls_mle`: least-squares and likelihood objectives plus a log-space optimiser,
  so the bias of unweighted least squares on count data can be shown rather than asserted (rung 1).
- `inference.optim`: multistart and cost surfaces (rung 2).
- `diagnostics`: log-log sensitivity spectrum, observed Fisher information, profile
  likelihood and profile intervals (rung 3).
- `inference.uncertainty`: Laplace intervals and a parametric bootstrap (rung 4).
- Notebooks `00`–`04`, one per rung, each closing with what it settles and what it does not.

### Notes

- The optimisation difficulty rung 2 was expected to demonstrate does not occur on this
  model: the likelihood over a complete epidemic has a single optimum, and twelve
  independent searches find it. The notebook reports that rather than manufacturing a
  harder problem, and the flat valley it does find is taken up by rung 3.
- Identifiability turns out to depend on the observation window. Over a complete epidemic
  both rates and `R0` are determined; over the exponential phase alone only `beta - gamma`
  is, and `R0` is unbounded. That is the opposite of the usual shorthand that `R0` is the
  rigid quantity.

## [0.1] - 2026-09-05

### Added

- `models.base`: the `SystemModel` / `ObservationModel` split and the `GenerativeModel`
  that composes them, keeping every rung of the ladder independent of the model it runs on.
- `models.sir`: deterministic SIR with a cumulative-incidence accumulator, so reported
  incidence is exact rather than approximated from prevalence.
- `observation`: Poisson and negative-binomial reporting, both with an underreporting
  fraction `rho`.
- `data`: seeded synthetic datasets with known ground truth, regenerated rather than stored.
- Tests for the conservation and monotonicity invariants, observation moments, and
  maximum-likelihood recovery of the generating parameters.

## [0.0] - 2026-09-05

### Added

- Repository scaffolding: `uv` project layout, ruff, pytest, mypy, pre-commit, GitHub Actions CI.
- MIT license, minimal README, this changelog.
