"""Observation models: the mean is easy, the variance is the point.

A predictor that gets the mean right and the spread wrong will produce confident
nonsense two rungs up, when those spreads turn into confidence intervals. So these
tests check the second moment as hard as the first, and check that the negative
binomial really does collapse to the Poisson in the limit -- if it did not, the
comparison between them later would be measuring an implementation difference
instead of a modelling one.
"""

import numpy as np
import pytest
from scipy.stats import poisson

from inferladder.observation import NegativeBinomialReporting, PoissonReporting

SIGNAL = np.array([10.0, 100.0, 1000.0])
DRAWS = 40_000


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def test_poisson_mean_is_the_reported_fraction_of_the_signal() -> None:
    model = PoissonReporting()
    assert np.allclose(model.mean(SIGNAL, np.array([0.3])), 0.3 * SIGNAL)


def test_poisson_samples_match_mean_and_variance(rng: np.random.Generator) -> None:
    model = PoissonReporting()
    phi = np.array([0.3])
    draws = np.array([model.sample(SIGNAL, phi, rng) for _ in range(DRAWS)])
    expected = model.mean(SIGNAL, phi)
    assert np.allclose(draws.mean(axis=0), expected, rtol=0.05)
    assert np.allclose(draws.var(axis=0), expected, rtol=0.05)


def test_poisson_log_prob_matches_the_reference_implementation() -> None:
    model = PoissonReporting()
    y = np.array([3, 30, 300])
    assert np.allclose(model.log_prob(y, SIGNAL, np.array([0.3])), poisson.logpmf(y, 0.3 * SIGNAL))


def test_negative_binomial_variance_exceeds_its_mean(rng: np.random.Generator) -> None:
    """``var = mean + mean^2 / k`` is the whole reason this model exists."""
    model = NegativeBinomialReporting()
    phi = np.array([0.3, 10.0])
    draws = np.array([model.sample(SIGNAL, phi, rng) for _ in range(DRAWS)])
    mean = model.mean(SIGNAL, phi)
    assert np.allclose(draws.mean(axis=0), mean, rtol=0.05)
    assert np.allclose(draws.var(axis=0), mean + mean**2 / 10.0, rtol=0.10)


def test_negative_binomial_approaches_poisson_as_dispersion_vanishes() -> None:
    y = np.array([3, 30, 300])
    negbin = NegativeBinomialReporting().log_prob(y, SIGNAL, np.array([0.3, 1e8]))
    pois = PoissonReporting().log_prob(y, SIGNAL, np.array([0.3]))
    assert np.allclose(negbin, pois, atol=1e-4)


def test_a_zero_signal_does_not_produce_a_negative_infinite_density() -> None:
    """Early in an epidemic the expected count underflows to zero; ``log 0`` is not a fit."""
    for model, phi in (
        (PoissonReporting(), np.array([0.3])),
        (
            NegativeBinomialReporting(),
            np.array([0.3, 10.0]),
        ),
    ):
        assert np.all(np.isfinite(model.log_prob(np.array([0]), np.array([0.0]), phi)))


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.5])
def test_poisson_rejects_an_impossible_reporting_fraction(rho: float) -> None:
    with pytest.raises(ValueError):
        PoissonReporting().mean(SIGNAL, np.array([rho]))


@pytest.mark.parametrize("phi", [np.array([0.3, 0.0]), np.array([0.3, -1.0]), np.array([0.3])])
def test_negative_binomial_rejects_invalid_parameters(phi: np.ndarray) -> None:
    with pytest.raises(ValueError):
        NegativeBinomialReporting().mean(SIGNAL, phi)
