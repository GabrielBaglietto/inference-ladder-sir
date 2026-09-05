# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
