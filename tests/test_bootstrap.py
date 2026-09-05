"""Parametric bootstrap, on the real model but at a size that stays cheap.

Kept small deliberately: this test certifies that the loop simulates, refits and
collects, not that a handful of replicates gives a good interval. The number of
replicates that makes an interval trustworthy is a question for the notebook.
"""

import numpy as np
import pytest

from inferladder.inference.uncertainty import parametric_bootstrap
from inferladder.models.base import GenerativeModel
from inferladder.models.sir import SIR
from inferladder.observation import PoissonReporting

TRUTH = np.array([0.35, 0.14, 0.80])
TIMES = np.arange(1.0, 81.0)


@pytest.fixture(scope="module")
def boot():
    model = GenerativeModel(SIR(population=1_000_000.0, i0=50.0), PoissonReporting())
    return model, parametric_bootstrap(model, TRUTH, TIMES, n_replicates=12, seed=2)


def test_every_replicate_is_collected_or_counted(boot) -> None:
    model, result = boot
    assert len(result.samples) + result.n_failed == 12
    assert result.samples.shape[1] == model.n_params
    assert result.param_names == model.param_names


def test_replicates_scatter_around_the_parameters_they_were_simulated_from(boot) -> None:
    _, result = boot
    assert np.allclose(result.samples.mean(axis=0), TRUTH, rtol=0.10)
    assert np.all(result.samples.std(axis=0) > 0)


def test_intervals_bracket_the_generating_parameters(boot) -> None:
    _, result = boot
    for index, name in enumerate(result.param_names):
        low, high = result.interval(name, level=0.99)
        assert low <= TRUTH[index] <= high, name


def test_derived_quantities_come_for_free(boot) -> None:
    """The reason bootstrap is worth its cost: R0 needs no delta-method algebra."""
    _, result = boot
    r0 = result.derived(lambda theta: theta[0] / theta[1])
    assert r0.shape == (len(result.samples),)
    assert np.median(r0) == pytest.approx(2.5, rel=0.15)


def test_the_bootstrap_is_reproducible() -> None:
    model = GenerativeModel(SIR(population=1_000_000.0, i0=50.0), PoissonReporting())
    first = parametric_bootstrap(model, TRUTH, TIMES, n_replicates=4, seed=5)
    second = parametric_bootstrap(model, TRUTH, TIMES, n_replicates=4, seed=5)
    assert np.allclose(first.samples, second.samples)
