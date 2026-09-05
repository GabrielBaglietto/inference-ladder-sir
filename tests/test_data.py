"""Synthetic data generation: reproducibility is the feature being tested.

Nothing is committed to ``data/``; every dataset is rebuilt from a seed. That is only
sound if the same seed really does give back the same numbers, so it is asserted
rather than assumed.
"""

import numpy as np
import pytest

from inferladder.data import DEFAULT_SEED, default_sir_dataset, default_sir_model, make_synthetic


def test_the_same_seed_reproduces_the_same_dataset() -> None:
    assert np.array_equal(default_sir_dataset(seed=7).y, default_sir_dataset(seed=7).y)


def test_different_seeds_give_different_noise() -> None:
    assert not np.array_equal(default_sir_dataset(seed=7).y, default_sir_dataset(seed=8).y)


def test_the_dataset_carries_its_ground_truth() -> None:
    dataset = default_sir_dataset()
    assert dataset.truth == {"beta": 0.35, "gamma": 0.14, "rho": 0.30, "k": 10.0}
    assert dataset.param_names == default_sir_model().param_names
    assert dataset.seed == DEFAULT_SEED


def test_counts_line_up_with_the_time_grid() -> None:
    dataset = default_sir_dataset(days=120)
    assert len(dataset) == 120
    assert dataset.t.shape == dataset.y.shape == dataset.signal_true.shape


def test_counts_are_non_negative_integers() -> None:
    dataset = default_sir_dataset()
    assert np.issubdtype(dataset.y.dtype, np.integer)
    assert dataset.y.min() >= 0


def test_the_epidemic_actually_happens() -> None:
    """A flat dataset would pass every other test here and teach nothing."""
    dataset = default_sir_dataset()
    assert dataset.y.max() > 100
    assert dataset.y.argmax() > 5


def test_observed_counts_track_the_reported_fraction_of_the_truth() -> None:
    """Noise should not move the total by much; a factor-of-two gap means a wiring bug."""
    dataset = default_sir_dataset()
    expected_total = 0.30 * dataset.signal_true.sum()
    assert dataset.y.sum() == pytest.approx(expected_total, rel=0.10)


def test_make_synthetic_respects_an_explicit_seed() -> None:
    model = default_sir_model()
    theta = np.array([0.35, 0.14, 0.30, 10.0])
    t = np.arange(1.0, 51.0)
    assert np.array_equal(
        make_synthetic(model, theta, t, seed=3).y, make_synthetic(model, theta, t, seed=3).y
    )
