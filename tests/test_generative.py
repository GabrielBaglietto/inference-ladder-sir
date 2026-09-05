"""Composition of a system model with an observation model.

``GenerativeModel`` is the only object that sees the whole parameter vector, so it is
where parameter bookkeeping goes wrong if it is going to. It is also the wall that
protects every optimiser and sampler above it: proposing impossible parameters must
return ``-inf``, never raise, or the first rung that explores the space will crash.
"""

import numpy as np
import pytest

from inferladder.data import default_sir_dataset, default_sir_model
from inferladder.models.base import GenerativeModel
from inferladder.models.sir import SIR
from inferladder.observation import PoissonReporting

TRUTH = np.array([0.35, 0.14, 0.30, 10.0])


@pytest.fixture
def model() -> GenerativeModel:
    return default_sir_model()


def test_parameters_are_concatenated_system_first(model: GenerativeModel) -> None:
    assert model.param_names == ("beta", "gamma", "rho", "k")
    assert model.n_params == 4


def test_split_separates_the_two_halves(model: GenerativeModel) -> None:
    theta_sys, phi = model.split(TRUTH)
    assert np.array_equal(theta_sys, [0.35, 0.14])
    assert np.array_equal(phi, [0.30, 10.0])


def test_a_wrongly_sized_parameter_vector_is_rejected(model: GenerativeModel) -> None:
    with pytest.raises(ValueError):
        model.split(np.array([0.35, 0.14]))


def test_expected_data_is_the_reported_fraction_of_incidence(model: GenerativeModel) -> None:
    t = np.arange(1.0, 61.0)
    signal = model.system.signal(TRUTH[:2], t)
    assert np.allclose(model.expected(TRUTH, t), 0.30 * signal)


def test_log_likelihood_is_finite_at_the_truth(model: GenerativeModel) -> None:
    dataset = default_sir_dataset()
    assert np.isfinite(model.log_likelihood(dataset.y, dataset.theta_true, dataset.t))


@pytest.mark.parametrize(
    "theta",
    [
        np.array([-0.35, 0.14, 0.30, 10.0]),
        np.array([0.35, -0.14, 0.30, 10.0]),
        np.array([0.35, 0.14, 1.50, 10.0]),
        np.array([0.35, 0.14, 0.30, -1.0]),
        np.array([0.35, 0.14, 0.30]),
    ],
    ids=["beta<0", "gamma<0", "rho>1", "k<0", "wrong-length"],
)
def test_impossible_parameters_return_minus_infinity_instead_of_raising(
    model: GenerativeModel, theta: np.ndarray
) -> None:
    """Optimisers walk into invalid regions constantly. They need a wall, not a crash."""
    dataset = default_sir_dataset()
    assert model.log_likelihood(dataset.y, theta, dataset.t) == -np.inf


@pytest.mark.parametrize("index,name", [(0, "beta"), (1, "gamma"), (2, "rho")])
def test_the_likelihood_falls_off_away_from_the_truth(index: int, name: str) -> None:
    """A cheap identifiability check: perturbing a parameter must cost likelihood.

    Deliberately uses a large, low-noise dataset and a coarse 25% perturbation, so it
    tests that the model is wired up rather than how sharp the peak is. Where the peak
    is *not* sharp is the subject of rung 3, not of a unit test.
    """
    model = GenerativeModel(SIR(population=1_000_000.0, i0=50.0), PoissonReporting())
    theta = np.array([0.35, 0.14, 0.80])
    t = np.arange(1.0, 161.0)
    y = model.sample(theta, t, np.random.default_rng(11))

    at_truth = model.log_likelihood(y, theta, t)
    for factor in (0.75, 1.25):
        moved = theta.copy()
        moved[index] *= factor
        assert model.log_likelihood(y, moved, t) < at_truth, f"{name} x{factor}"
