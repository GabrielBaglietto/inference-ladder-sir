"""Synthetic data with known ground truth.

Everything on the ladder is validated here first, against data whose parameters we
chose. That is the only setting in which "did the method recover the answer?" is a
question with a checkable answer -- on real data every method returns *something*,
and nothing tells you it is wrong.

Datasets are regenerated from a seed rather than stored, so results stay reproducible
without committing files.
"""

from dataclasses import dataclass, field

import numpy as np

from inferladder.models.base import CountArray, FloatArray, GenerativeModel
from inferladder.models.sir import SIR
from inferladder.observation import NegativeBinomialReporting

DEFAULT_SEED = 20260905


@dataclass(frozen=True)
class Dataset:
    """Observed counts together with the truth that produced them.

    ``theta_true`` is present precisely because this is synthetic. Any code that
    reads it outside of a validation check is cheating, and will not survive the
    move to real data.
    """

    t: FloatArray
    y: CountArray
    theta_true: FloatArray
    param_names: tuple[str, ...]
    seed: int
    signal_true: FloatArray = field(repr=False)

    @property
    def truth(self) -> dict[str, float]:
        """Ground-truth parameters as a name-keyed mapping, for readable assertions."""
        return dict(zip(self.param_names, map(float, self.theta_true), strict=True))

    def __len__(self) -> int:
        return int(self.y.size)


def make_synthetic(
    model: GenerativeModel,
    theta: FloatArray,
    t: FloatArray,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Draw one dataset from ``model`` at parameters ``theta``."""
    theta = np.asarray(theta, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    rng = np.random.default_rng(seed)

    theta_sys, _ = model.split(theta)
    signal = model.system.signal(theta_sys, t)
    y = model.sample(theta, t, rng)

    return Dataset(
        t=t,
        y=y,
        theta_true=theta,
        param_names=model.param_names,
        seed=seed,
        signal_true=signal,
    )


def default_sir_model() -> GenerativeModel:
    """The reference setup used throughout the ladder.

    A negative-binomial reporting model sits on top of the SIR, so that the very
    first synthetic dataset already has the overdispersion that real data has. A
    Poisson-generated dataset would make the early rungs look better than they are.
    """
    return GenerativeModel(system=SIR(), observation=NegativeBinomialReporting())


def default_sir_dataset(seed: int = DEFAULT_SEED, days: int = 160) -> Dataset:
    """The reference synthetic epidemic: ``R0 = 2.5``, 30% of cases reported.

    Parameters are chosen so the outbreak takes off and burns out inside the window.
    A curve that is still exponential at the end of the record is a much harder and
    much less identifiable problem, and it is not the one the early rungs are about.
    """
    model = default_sir_model()
    theta = np.array([0.35, 0.14, 0.30, 10.0], dtype=np.float64)
    t = np.arange(1.0, days + 1.0)
    return make_synthetic(model, theta, t, seed=seed)
