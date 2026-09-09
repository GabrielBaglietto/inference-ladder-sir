"""Rung 5: the posterior sampler.

Runs at a size that exercises the machinery rather than one that yields a trustworthy
posterior -- short chains on a short series. What a *usable* posterior costs is the
notebook's business; what this file certifies is that the black-box likelihood reaches
PyMC intact, that the priors are the ones declared, and that convergence is reported
rather than assumed.

Skipped entirely when the optional `bayes` group is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("pymc", reason="the optional 'bayes' dependency group is not installed")

from inferladder.data import make_synthetic  # noqa: E402
from inferladder.inference.bayes import (  # noqa: E402
    ConvergenceReport,
    Priors,
    build_model,
    check_convergence,
    posterior_samples,
    sample_posterior,
)
from inferladder.models.base import GenerativeModel  # noqa: E402
from inferladder.models.sir import SIR  # noqa: E402
from inferladder.observation import NegativeBinomialReporting  # noqa: E402

TRUTH = np.array([0.35, 0.14, 0.30, 10.0])
TIMES = np.arange(1.0, 61.0)


@pytest.fixture(scope="module")
def small_problem():
    model = GenerativeModel(SIR(population=100_000.0), NegativeBinomialReporting())
    dataset = make_synthetic(model, TRUTH, TIMES, seed=7)
    return model, dataset


@pytest.fixture(scope="module")
def idata(small_problem):
    model, dataset = small_problem
    return sample_posterior(
        model, dataset.y, dataset.t, draws=250, tune=250, chains=2, cores=1, seed=3
    )


def test_the_model_exposes_the_declared_parameters(small_problem) -> None:
    model, dataset = small_problem
    pymc_model = build_model(model, dataset.y, dataset.t)
    names = {variable.name for variable in pymc_model.free_RVs}
    assert names == {"beta", "gamma", "rho", "k"}


def test_r0_is_recorded_as_a_derived_quantity(small_problem) -> None:
    """R0 is not a parameter, and reconstructing it afterwards would lose the correlation."""
    model, dataset = small_problem
    pymc_model = build_model(model, dataset.y, dataset.t)
    assert "R0" in {variable.name for variable in pymc_model.deterministics}


def test_a_model_with_different_parameters_is_refused(small_problem) -> None:
    """These priors are written for (beta, gamma, rho, k); silently reusing them would lie."""
    from inferladder.observation import PoissonReporting

    _, dataset = small_problem
    poisson_model = GenerativeModel(SIR(), PoissonReporting())
    with pytest.raises(ValueError, match="beta, gamma, rho, k"):
        build_model(poisson_model, dataset.y, dataset.t)


def test_an_unknown_sampler_is_refused(small_problem) -> None:
    model, dataset = small_problem
    with pytest.raises(ValueError, match="unknown sampler"):
        sample_posterior(model, dataset.y, dataset.t, sampler="nuts", draws=5, tune=5, chains=1)


def test_sampling_returns_draws_for_every_parameter(idata, small_problem) -> None:
    model, _ = small_problem
    samples = posterior_samples(idata, model.param_names)
    assert samples.shape == (500, 4)
    assert np.all(np.isfinite(samples))


def test_the_draws_respect_the_support_of_the_priors(idata) -> None:
    """rho is a fraction and the rest are positive; a sampler escaping that is broken."""
    posterior = idata.posterior
    assert float(posterior["rho"].min()) > 0.0
    assert float(posterior["rho"].max()) < 1.0
    for name in ("beta", "gamma", "k"):
        assert float(posterior[name].min()) > 0.0


def test_convergence_is_reported_rather_than_assumed(idata) -> None:
    report = check_convergence(idata, var_names=["beta", "gamma", "rho", "k"])
    assert isinstance(report, ConvergenceReport)
    assert report.max_r_hat >= 1.0
    assert set(report.per_parameter) == {"beta", "gamma", "rho", "k"}


def test_the_convergence_threshold_is_strict() -> None:
    """A stuck chain understates the interval, so 'roughly converged' is not a category."""
    assert ConvergenceReport(max_r_hat=1.005, min_ess_bulk=500, n_divergences=0).converged
    assert not ConvergenceReport(max_r_hat=1.05, min_ess_bulk=500, n_divergences=0).converged
    assert not ConvergenceReport(max_r_hat=1.0, min_ess_bulk=50, n_divergences=0).converged


def test_priors_are_data_not_hidden_constants() -> None:
    custom = Priors(rho_alpha=5.0, rho_beta=5.0)
    assert custom.rho_alpha == 5.0
    assert Priors().rho_alpha == 2.0


def test_sampling_is_reproducible(small_problem) -> None:
    model, dataset = small_problem
    kwargs = dict(draws=60, tune=60, chains=1, cores=1, seed=11)
    first = posterior_samples(
        sample_posterior(model, dataset.y, dataset.t, **kwargs), model.param_names
    )
    second = posterior_samples(
        sample_posterior(model, dataset.y, dataset.t, **kwargs), model.param_names
    )
    assert np.allclose(first, second)
