"""Invariants of the SIR system model.

These are the properties that must hold for *any* parameters, so they are the ones
worth asserting: a solver tolerance that drifts, a sign error in the flux, or a
mix-up between prevalence and incidence all show up here rather than as a slightly
worse fit three rungs later.
"""

import numpy as np
import pytest

from inferladder.models.sir import SIR

THETA = np.array([0.35, 0.14])
TIMES = np.arange(1.0, 161.0)


@pytest.fixture
def sir() -> SIR:
    return SIR(population=100_000.0, i0=10.0)


def test_population_is_conserved(sir: SIR) -> None:
    states = sir.simulate(THETA, TIMES)
    assert np.allclose(states.sum(axis=1), sir.population, rtol=0, atol=1e-6)


def test_states_stay_non_negative(sir: SIR) -> None:
    states = sir.simulate(THETA, TIMES)
    assert states.min() > -1e-9


def test_susceptibles_never_increase(sir: SIR) -> None:
    susceptible = sir.simulate(THETA, TIMES)[:, 0]
    assert np.all(np.diff(susceptible) <= 1e-9)


def test_removed_never_decrease(sir: SIR) -> None:
    removed = sir.simulate(THETA, TIMES)[:, 2]
    assert np.all(np.diff(removed) >= -1e-9)


def test_simulate_shape_matches_grid(sir: SIR) -> None:
    assert sir.simulate(THETA, TIMES).shape == (TIMES.size, len(sir.state_names))


def test_incidence_is_non_negative_and_aligned_with_the_grid(sir: SIR) -> None:
    signal = sir.signal(THETA, TIMES)
    assert signal.shape == TIMES.shape
    assert np.all(signal >= 0.0)


def test_incidence_accounts_for_every_departure_from_susceptible(sir: SIR) -> None:
    """Total incidence must equal the drop in S: nobody is infected twice, nobody vanishes.

    This is the test that catches prevalence being reported as incidence -- the two
    agree in shape and disagree in total.
    """
    total_infections = sir.signal(THETA, TIMES).sum()
    final_susceptible = sir.simulate(THETA, TIMES)[-1, 0]
    initial_susceptible = sir.initial_state()[0]
    assert total_infections == pytest.approx(initial_susceptible - final_susceptible, rel=1e-6)


def test_epidemic_takes_off_when_r0_exceeds_one(sir: SIR) -> None:
    growing = sir.simulate(np.array([0.35, 0.14]), TIMES)[:, 1]
    assert growing.max() > sir.i0 * 10


def test_epidemic_dies_out_when_r0_is_below_one(sir: SIR) -> None:
    fading = sir.simulate(np.array([0.10, 0.20]), TIMES)[:, 1]
    assert np.all(np.diff(fading) < 0)


def test_basic_reproduction_number(sir: SIR) -> None:
    assert sir.basic_reproduction_number(THETA) == pytest.approx(2.5)


def test_simulation_is_deterministic(sir: SIR) -> None:
    assert np.array_equal(sir.simulate(THETA, TIMES), sir.simulate(THETA, TIMES))


@pytest.mark.parametrize("theta", [np.array([-0.1, 0.14]), np.array([0.35, 0.0]), np.array([0.35])])
def test_invalid_parameters_are_rejected(sir: SIR, theta: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sir.simulate(theta, TIMES)


@pytest.mark.parametrize(
    "times",
    [
        np.array([1.0]),
        np.array([1.0, 2.0, 4.0]),
        np.array([3.0, 2.0, 1.0]),
    ],
    ids=["too-short", "ragged", "decreasing"],
)
def test_unusable_time_grids_are_rejected(sir: SIR, times: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sir.signal(THETA, times)


@pytest.mark.parametrize("kwargs", [{"population": 0.0}, {"i0": 0.0}, {"i0": 1e9}])
def test_invalid_configurations_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SIR(**kwargs)
