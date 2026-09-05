"""Observation models: from an expected signal to a distribution over counts.

Two things happen between an epidemic and a case count, and both live here.

*Underreporting.* Only a fraction ``rho`` of infections is ever confirmed. This is
the parameter that makes the problem interesting: scaling ``rho`` down and the
epidemic size up leaves the expected counts almost unchanged, so ``rho`` is entangled
with the system parameters and cannot be read off the curve alone.

*Noise.* Poisson is the honest starting point -- counting independent events -- and
it is almost always wrong for real surveillance data, whose variance far exceeds its
mean because of weekday effects, batched reporting and clustered transmission. The
negative binomial adds one dispersion parameter to absorb that. Rung 4 of the ladder
is where the difference stops being cosmetic and starts changing the uncertainty.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.stats import nbinom, poisson

from inferladder.models.base import CountArray, FloatArray

_TINY = 1e-12
"""Floor for the expected count, so that a zero signal does not produce ``log(0)``."""


@dataclass(frozen=True)
class PoissonReporting:
    """``y_t ~ Poisson(rho * signal_t)``.

    The minimal observation model: pure counting noise on a fraction of the truth.
    Its variance is locked to its mean, which is exactly the assumption real
    surveillance data violates.
    """

    param_names: ClassVar[tuple[str, ...]] = ("rho",)

    def mean(self, signal: FloatArray, phi: FloatArray) -> FloatArray:
        rho = _unpack_rho(phi)
        return np.maximum(rho * np.asarray(signal, dtype=np.float64), _TINY)

    def sample(self, signal: FloatArray, phi: FloatArray, rng: np.random.Generator) -> CountArray:
        return rng.poisson(self.mean(signal, phi)).astype(np.int64)

    def log_prob(self, y: CountArray, signal: FloatArray, phi: FloatArray) -> FloatArray:
        result: FloatArray = poisson.logpmf(y, self.mean(signal, phi))
        return result


@dataclass(frozen=True)
class NegativeBinomialReporting:
    """``y_t ~ NegBinomial(mean = rho * signal_t, dispersion = k)``.

    Parameterised so that ``var = mean + mean^2 / k``: large ``k`` collapses to
    Poisson, small ``k`` means heavy overdispersion. ``k`` is carried as a free
    parameter rather than fixed, because pinning it at a guess quietly hands the fit
    a confidence it has not earned.
    """

    param_names: ClassVar[tuple[str, ...]] = ("rho", "k")

    def mean(self, signal: FloatArray, phi: FloatArray) -> FloatArray:
        rho, _ = _unpack_rho_k(phi)
        return np.maximum(rho * np.asarray(signal, dtype=np.float64), _TINY)

    def sample(self, signal: FloatArray, phi: FloatArray, rng: np.random.Generator) -> CountArray:
        _, k = _unpack_rho_k(phi)
        mean = self.mean(signal, phi)
        # Gamma-Poisson mixture: the standard construction, and it keeps the
        # parameterisation identical to the one used in log_prob.
        rate = rng.gamma(shape=k, scale=mean / k)
        return rng.poisson(rate).astype(np.int64)

    def log_prob(self, y: CountArray, signal: FloatArray, phi: FloatArray) -> FloatArray:
        _, k = _unpack_rho_k(phi)
        mean = self.mean(signal, phi)
        result: FloatArray = nbinom.logpmf(y, k, k / (k + mean))
        return result


def _unpack_rho(phi: FloatArray) -> float:
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (1,):
        raise ValueError(f"expected (rho,), got shape {phi.shape}")
    rho = float(phi[0])
    if not 0 < rho <= 1:
        raise ValueError("rho must lie in (0, 1]")
    return rho


def _unpack_rho_k(phi: FloatArray) -> tuple[float, float]:
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (2,):
        raise ValueError(f"expected (rho, k), got shape {phi.shape}")
    rho, k = float(phi[0]), float(phi[1])
    if not 0 < rho <= 1:
        raise ValueError("rho must lie in (0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")
    return rho, k
