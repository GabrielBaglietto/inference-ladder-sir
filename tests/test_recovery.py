"""Does maximum likelihood find the parameters that generated the data?

This is the one check that cannot be made on real data, and it is the reason the
ladder starts synthetic. It is *not* a preview of the ladder: rungs 0 to 4 are about
how you get here, which estimator to trust and how wide the uncertainty is. This test
only certifies that the generative model is self-consistent -- that the truth sits
where the likelihood is maximised, so that later rungs are climbing a real hill.

The setting is deliberately generous: a large population, Poisson noise and 80% of
cases reported. Recovery on the harder default dataset is measurably worse, and that
degradation is a finding to be explored in a notebook, not a tolerance to be widened
here until the test passes.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from inferladder.models.base import GenerativeModel
from inferladder.models.sir import SIR
from inferladder.observation import PoissonReporting

TRUTH = np.array([0.35, 0.14, 0.80])
TIMES = np.arange(1.0, 161.0)
TOLERANCE = 0.05


@pytest.fixture(scope="module")
def fitted() -> tuple[np.ndarray, GenerativeModel]:
    model = GenerativeModel(SIR(population=1_000_000.0, i0=50.0), PoissonReporting())
    y = model.sample(TRUTH, TIMES, np.random.default_rng(1))

    # Optimise in log-space: all three parameters are positive, and the optimiser
    # would otherwise spend its time proposing negative rates.
    result = minimize(
        lambda u: -model.log_likelihood(y, np.exp(u), TIMES),
        np.log(TRUTH) + 0.2,
        method="Nelder-Mead",
    )
    assert result.success, result.message
    return np.exp(result.x), model


@pytest.mark.parametrize("index,name", [(0, "beta"), (1, "gamma"), (2, "rho")])
def test_maximum_likelihood_recovers_each_parameter(
    fitted: tuple[np.ndarray, GenerativeModel], index: int, name: str
) -> None:
    estimate, _ = fitted
    assert estimate[index] == pytest.approx(TRUTH[index], rel=TOLERANCE), name


def test_maximum_likelihood_recovers_the_reproduction_number(
    fitted: tuple[np.ndarray, GenerativeModel],
) -> None:
    estimate, _ = fitted
    assert estimate[0] / estimate[1] == pytest.approx(TRUTH[0] / TRUTH[1], rel=TOLERANCE)


def test_the_fit_beats_the_truth_on_its_own_data(
    fitted: tuple[np.ndarray, GenerativeModel],
) -> None:
    """The maximum-likelihood estimate is not the truth, and should not be.

    On a finite noisy sample the fit necessarily explains that sample at least as
    well as the parameters that generated it. A fit that scored *worse* than the
    truth would mean the optimiser had not converged.
    """
    estimate, model = fitted
    y = model.sample(TRUTH, TIMES, np.random.default_rng(1))
    assert model.log_likelihood(y, estimate, TIMES) >= model.log_likelihood(y, TRUTH, TIMES)
