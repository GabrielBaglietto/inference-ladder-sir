"""Rungs 1 and 2: fitting machinery, tested on objectives with known minima.

Analytic objectives on purpose. These tests are about the optimisation plumbing --
log-space transform, bookkeeping, multistart, grid evaluation -- and running them
against the ODE would make them slow and would confound plumbing bugs with modelling
questions that belong in the notebooks.
"""

import numpy as np
import pytest

from inferladder.inference.nls_mle import fit, negative_log_likelihood, sum_of_squares
from inferladder.inference.optim import cost_surface, multistart, random_starts

NAMES = ("a", "b")
MINIMUM = np.array([2.0, 5.0])


def quadratic(theta: np.ndarray) -> float:
    """A bowl in log-parameter space, minimised exactly at MINIMUM."""
    return float(np.sum((np.log(theta) - np.log(MINIMUM)) ** 2))


def test_fit_finds_a_known_minimum() -> None:
    estimate = fit(quadratic, np.array([1.0, 1.0]), NAMES)
    assert estimate.success
    assert np.allclose(estimate.theta, MINIMUM, rtol=1e-3)


def test_the_estimate_is_readable_by_name() -> None:
    estimate = fit(quadratic, np.array([1.0, 1.0]), NAMES)
    assert estimate["a"] == pytest.approx(2.0, rel=1e-3)
    assert set(estimate.as_dict) == {"a", "b"}


def test_a_non_positive_start_is_rejected() -> None:
    """Optimisation happens in log space, so a zero or negative start has no logarithm."""
    with pytest.raises(ValueError, match="log space"):
        fit(quadratic, np.array([0.0, 1.0]), NAMES)


def test_mismatched_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        fit(quadratic, np.array([1.0, 1.0]), ("a", "b", "c"))


def test_multistart_returns_the_best_run_and_all_of_them() -> None:
    starts = random_starts(MINIMUM, 5, spread=0.8, seed=0)
    best, runs = multistart(quadratic, starts, NAMES)
    assert len(runs) == 5
    assert best.objective_value == min(run.objective_value for run in runs)
    assert np.allclose(best.theta, MINIMUM, rtol=1e-3)


def test_multistart_needs_at_least_one_start() -> None:
    with pytest.raises(ValueError):
        multistart(quadratic, [], NAMES)


def test_random_starts_scatter_around_the_centre_within_the_spread() -> None:
    starts = np.array(random_starts(MINIMUM, 200, spread=0.5, seed=1))
    ratios = np.log(starts / MINIMUM)
    assert np.all(np.abs(ratios) <= 0.5 + 1e-9)
    assert ratios.std() > 0.1


def test_random_starts_are_reproducible() -> None:
    assert np.allclose(random_starts(MINIMUM, 5, seed=3), random_starts(MINIMUM, 5, seed=3))


def test_cost_surface_brackets_the_minimum() -> None:
    surface = cost_surface(quadratic, MINIMUM * 1.1, (0, 1), NAMES, factor=1.5, resolution=21)
    assert surface.values.shape == (21, 21)
    x_min, y_min = surface.minimum
    assert x_min == pytest.approx(2.0, rel=0.1)
    assert y_min == pytest.approx(5.0, rel=0.1)
    assert (surface.x_name, surface.y_name) == NAMES


@pytest.mark.parametrize(
    "kwargs", [{"indices": (0, 0)}, {"factor": 1.0}], ids=["same-parameter", "no-width"]
)
def test_a_degenerate_surface_request_is_rejected(kwargs: dict) -> None:
    call = {"indices": (0, 1), "factor": 1.5, **kwargs}
    with pytest.raises(ValueError):
        cost_surface(quadratic, MINIMUM, call["indices"], NAMES, factor=call["factor"])


def test_objectives_are_built_against_the_model() -> None:
    """Both objective builders must survive parameters the model rejects."""
    from inferladder import default_sir_dataset, default_sir_model

    dataset = default_sir_dataset(days=30)
    model = default_sir_model()
    impossible = np.array([0.35, 0.14, 1.5, 10.0])

    assert negative_log_likelihood(model, dataset.y, dataset.t)(impossible) == np.inf
    assert sum_of_squares(model, dataset.y, dataset.t)(impossible) == np.inf
