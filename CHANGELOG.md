# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3] - 2026-09-09

### Added

- `inference.bayes`: the SIR likelihood wrapped as a black-box PyTensor operation, stated
  priors, posterior sampling and a convergence report that is checked before the numbers
  are read. PyMC and ArviZ live in an optional `bayes` dependency group.
- Notebook `05`: the reference posterior, its agreement with Laplace, and a decision curve
  obtained by pushing the joint posterior through a policy question.

### Notes

- **Not NUTS.** NUTS needs the gradient of the log-posterior, and the dynamics are
  integrated by `scipy`, which automatic differentiation cannot see through. Supplying
  gradients would mean a second implementation of the same model in a differentiable
  framework, which is the drift the model-agnostic design exists to prevent. The sampler
  is `DEMetropolisZ`, which learns proposal directions from the chain's own history.
- That choice is not free, and notebook 05 shows the cost: a componentwise Metropolis on
  the same posterior retains an effective sample size in the single digits out of 3200
  draws, because `beta` and `gamma` correlate at 0.99. The sloppiness diagnosed at rung 3
  breaks samplers that ignore it.
- Laplace and the posterior agree closely on `beta`, `gamma` and `rho`. The reason to
  climb this rung is the joint distribution and the priors, not tighter intervals.

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
