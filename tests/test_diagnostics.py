"""Rungs 3 and 4: identifiability diagnostics and intervals.

Checked against functions whose derivatives are known on paper, so a failure points at
the finite-difference machinery rather than at the epidemic model.
"""

import numpy as np
import pytest

from inferladder.diagnostics import (
    log_jacobian,
    observed_fisher,
    profile_interval,
    profile_likelihood,
    sensitivity_spectrum,
)
from inferladder.inference.uncertainty import (
    laplace_covariance,
    laplace_intervals,
    normal_approximation_intervals,
)

NAMES = ("a", "b")


def power_law(theta: np.ndarray) -> np.ndarray:
    """``f = a^2 * b``, whose log-log derivatives are exactly ``[2, 1]``."""
    return np.array([theta[0] ** 2 * theta[1]])


def redundant_power_law(theta: np.ndarray) -> np.ndarray:
    """Two predictions that are both functions of ``a^2 b`` alone.

    Rank one by construction: moving along ``a^2 b = const`` changes neither output,
    so the decomposition must report a null direction.
    """
    combination = theta[0] ** 2 * theta[1]
    return np.array([combination, combination**2])


def log_space_bowl(theta: np.ndarray) -> float:
    """``0.5 * sum(log theta ^ 2)``: Hessian in log space is the identity."""
    return float(0.5 * np.sum(np.log(theta) ** 2))


def test_the_log_jacobian_reproduces_known_elasticities() -> None:
    jacobian = log_jacobian(power_law, np.array([3.0, 7.0]))
    assert np.allclose(jacobian, [[2.0, 1.0]], atol=1e-6)


def test_the_spectrum_finds_the_direction_the_data_cannot_see() -> None:
    """``a^2 b`` is unchanged along ``a^2 b = const``, so one direction is invisible."""
    spectrum = sensitivity_spectrum(redundant_power_law, np.array([3.0, 7.0]), NAMES)
    assert spectrum.singular_values[-1] == pytest.approx(0.0, abs=1e-6)
    assert spectrum.condition_number == np.inf

    # The invisible direction is the one holding ``a^2 b`` fixed: d(log a) = -2 d(log b).
    sloppy = spectrum.sloppiest()
    assert sloppy["a"] / sloppy["b"] == pytest.approx(-0.5, rel=1e-3)


def test_the_stiff_direction_carries_most_of_the_sensitivity() -> None:
    spectrum = sensitivity_spectrum(redundant_power_law, np.array([3.0, 7.0]), NAMES)
    stiff = spectrum.stiffest()
    assert abs(stiff["a"]) > abs(stiff["b"])


def test_a_spectrum_with_fewer_predictions_than_parameters_is_refused() -> None:
    """One prediction cannot rank three parameters, and pretending otherwise is worse."""
    with pytest.raises(ValueError, match="rank-deficient"):
        sensitivity_spectrum(power_law, np.array([3.0, 7.0]), NAMES)


def test_the_observed_fisher_matches_a_known_hessian() -> None:
    hessian = observed_fisher(log_space_bowl, np.array([2.0, 5.0]))
    assert np.allclose(hessian, np.eye(2), atol=1e-3)


def test_the_hessian_is_symmetric() -> None:
    def tilted(theta: np.ndarray) -> float:
        u = np.log(theta)
        return float(0.5 * (u[0] ** 2 + 3 * u[1] ** 2) + 0.7 * u[0] * u[1])

    hessian = observed_fisher(tilted, np.array([2.0, 5.0]))
    assert np.allclose(hessian, hessian.T, atol=1e-12)
    assert hessian[0, 1] == pytest.approx(0.7, abs=1e-3)


def test_a_profile_is_minimised_at_the_optimum() -> None:
    grid = np.linspace(0.5, 2.0, 9)
    profile = profile_likelihood(log_space_bowl, np.array([1.0, 1.0]), 0, grid, NAMES)
    assert profile.name == "a"
    assert profile.values.shape == (9,)
    assert profile.delta.min() == pytest.approx(0.0)
    assert grid[int(np.argmin(profile.objective))] == pytest.approx(1.0, abs=0.2)


def test_profiling_needs_something_left_to_re_optimise() -> None:
    with pytest.raises(ValueError):
        profile_likelihood(log_space_bowl, np.array([1.0]), 0, [1.0], ("a",))


def test_a_profile_interval_brackets_the_optimum() -> None:
    grid = np.linspace(0.2, 5.0, 25)
    profile = profile_likelihood(log_space_bowl, np.array([1.0, 1.0]), 0, grid, NAMES)
    low, high = profile_interval(profile)
    assert low < 1.0 < high


def test_an_unbounded_profile_reports_an_infinite_limit() -> None:
    """A grid that never rises above the threshold must not report its own edge."""

    def flat(theta: np.ndarray) -> float:
        return float(np.log(theta[1]) ** 2)  # first parameter does nothing at all

    grid = np.linspace(0.5, 2.0, 7)
    profile = profile_likelihood(flat, np.array([1.0, 1.0]), 0, grid, NAMES)
    assert profile_interval(profile) == (-np.inf, np.inf)


def test_laplace_refuses_a_hessian_with_no_curvature() -> None:
    """A non-positive-definite Hessian is a finding about the model, not a nuisance."""
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        laplace_covariance(np.array([[1.0, 0.0], [0.0, -0.5]]))


def test_laplace_intervals_are_asymmetric_on_the_original_scale() -> None:
    """Computed in log space and mapped back, so a positive quantity stays positive."""
    intervals = laplace_intervals(np.array([2.0, 5.0]), np.eye(2), NAMES, level=0.95)
    low, high = intervals["a"]
    assert 0 < low < 2.0 < high
    assert (high - 2.0) > (2.0 - low)


def test_a_sharper_likelihood_gives_a_tighter_interval() -> None:
    loose = laplace_intervals(np.array([2.0, 5.0]), np.eye(2), NAMES)["a"]
    sharp = laplace_intervals(np.array([2.0, 5.0]), 100 * np.eye(2), NAMES)["a"]
    assert (sharp[1] - sharp[0]) < (loose[1] - loose[0])


def test_the_normal_summary_matches_a_normal_sample() -> None:
    sample = np.random.default_rng(0).normal(10.0, 2.0, size=5_000)
    low, high = normal_approximation_intervals(sample)
    assert low == pytest.approx(10.0 - 1.96 * 2.0, abs=0.2)
    assert high == pytest.approx(10.0 + 1.96 * 2.0, abs=0.2)
