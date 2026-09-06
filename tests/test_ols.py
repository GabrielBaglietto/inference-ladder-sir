"""Rung 0: the log-linear growth-rate fit."""

import numpy as np
import pytest

from inferladder.inference.ols import exponential_growth_rate


def _exponential_counts(
    rate: float = 0.21, n: int = 30, seed: int = 0, intercept: float = 3.0
) -> tuple:
    """Poisson counts on an exponential mean.

    ``intercept`` sets how large the counts are, and it matters: see
    ``test_small_counts_bias_the_slope_upward`` below.
    """
    t = np.arange(1.0, n + 1.0)
    rng = np.random.default_rng(seed)
    y = rng.poisson(np.exp(intercept + rate * t)).astype(np.int64)
    return t, y


def test_it_recovers_a_known_growth_rate() -> None:
    t, y = _exponential_counts(rate=0.21)
    fit = exponential_growth_rate(t, y, window=(1.0, 30.0))
    assert fit.slope == pytest.approx(0.21, rel=0.05)


def test_the_interval_covers_the_truth() -> None:
    t, y = _exponential_counts(rate=0.21)
    low, high = exponential_growth_rate(t, y, window=(1.0, 30.0)).confidence_interval()
    assert low <= 0.21 <= high


def test_a_wider_level_gives_a_wider_interval() -> None:
    t, y = _exponential_counts()
    fit = exponential_growth_rate(t, y, window=(1.0, 30.0))
    narrow = fit.confidence_interval(0.80)
    wide = fit.confidence_interval(0.99)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_doubling_time_matches_the_growth_rate() -> None:
    t, y = _exponential_counts(rate=np.log(2.0), n=12)
    assert exponential_growth_rate(t, y, window=(1.0, 30.0)).doubling_time() == pytest.approx(
        1.0, rel=0.1
    )


def test_a_shrinking_epidemic_has_no_doubling_time() -> None:
    t, y = _exponential_counts(rate=-0.05)
    assert exponential_growth_rate(t, y, window=(1.0, 30.0)).doubling_time() == np.inf


def test_zero_counts_are_dropped_rather_than_shifted() -> None:
    """A pseudo-count would distort exactly the smallest, highest-leverage points."""
    t, y = _exponential_counts()
    y = y.copy()
    y[:3] = 0
    fit = exponential_growth_rate(t, y, window=(1.0, 30.0))
    assert fit.n_points == 27


@pytest.mark.parametrize("window", [(20.0, 5.0), (5.0, 5.0)])
def test_a_non_increasing_window_is_rejected(window: tuple[float, float]) -> None:
    t, y = _exponential_counts()
    with pytest.raises(ValueError):
        exponential_growth_rate(t, y, window=window)


def test_a_window_with_too_few_points_is_rejected() -> None:
    t, y = _exponential_counts()
    with pytest.raises(ValueError, match="usable points"):
        exponential_growth_rate(t, y, window=(1.0, 2.0))


def test_mismatched_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        exponential_growth_rate(np.arange(5.0), np.arange(4), window=(0.0, 4.0))


def test_small_counts_bias_the_slope_upward() -> None:
    """Regressing ``log(y)`` is biased when the counts are small, and biased upward.

    ``E[log y] < log E[y]`` by Jensen, and the gap is worst where the counts are
    smallest -- which on an exponential curve is the left-hand end, where the fit has
    the most leverage. The result is a growth rate that reads high. Averaged over
    many draws the effect is small but systematic, and it does not shrink with more
    data at the same count size; only larger counts remove it.

    This is a property of the estimator, not a defect of the implementation, so it is
    pinned down here rather than corrected.
    """

    def mean_slope(intercept: float) -> float:
        slopes = [
            exponential_growth_rate(
                *_exponential_counts(seed=seed, intercept=intercept), window=(1.0, 30.0)
            ).slope
            for seed in range(40)
        ]
        return float(np.mean(slopes))

    tiny_counts = mean_slope(1.0)
    large_counts = mean_slope(5.0)

    assert tiny_counts > 0.21
    assert large_counts == pytest.approx(0.21, abs=0.001)
    assert abs(tiny_counts - 0.21) > abs(large_counts - 0.21)
