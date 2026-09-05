"""The seams that keep the ladder model-agnostic.

Every rung of the ladder is written once and re-run against different mechanistic
models. That only works if nothing above this module knows what an SIR is. Two
interfaces make that possible, and the split between them is deliberate:

``SystemModel``
    Parameters to latent trajectory. Deterministic, no noise. It also exposes the
    *signal*: the quantity an instrument would try to measure. For an epidemic that
    is incidence -- new infections per reporting period -- which is a flux, not a
    state variable. Conflating the two is a classic source of bias.

``ObservationModel``
    Signal to a *distribution* over data. Not to a number. This is where reporting
    fraction, delays and overdispersion live.

Keeping them apart is what turns a curve fit into an identifiability problem: the
same epidemic reported at half the rate looks like a smaller epidemic that infected
twice as many people, and only a prior or an independent measurement can tell the
two apart. A single black box mapping parameters to data would hide that.

``GenerativeModel`` composes the two into the object the inference code consumes.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
CountArray = npt.NDArray[np.int64]


def uniform_step(t: FloatArray) -> float:
    """Return the spacing of a uniform, increasing time grid.

    Reporting periods are assumed regular (daily counts, weekly counts). Ragged
    grids are a real complication -- they change what a "case count" even means --
    so they are rejected here rather than silently averaged over.
    """
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"expected a 1-d grid of at least 2 times, got shape {t.shape}")
    steps = np.diff(t)
    if np.any(steps <= 0):
        raise ValueError("time grid must be strictly increasing")
    if not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
        raise ValueError("time grid must be uniformly spaced")
    return float(steps[0])


@runtime_checkable
class SystemModel(Protocol):
    """Deterministic map from parameters to latent states and to an observable signal."""

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the system parameters, in the order ``simulate`` expects them."""

    @property
    def state_names(self) -> tuple[str, ...]:
        """Names of the latent state variables, in column order."""

    def simulate(self, theta: FloatArray, t: FloatArray) -> FloatArray:
        """Latent states at each time in ``t``, shaped ``(len(t), len(state_names))``."""

    def signal(self, theta: FloatArray, t: FloatArray) -> FloatArray:
        """Expected observable at each time in ``t``, shaped ``(len(t),)``.

        Times are the *end* of each reporting period, so ``signal[i]`` covers the
        interval ending at ``t[i]``. Element 0 is a full period, not a partial one.
        """


@runtime_checkable
class ObservationModel(Protocol):
    """Map from an expected signal to a distribution over observed data."""

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the observation parameters, in the order the methods expect them."""

    def mean(self, signal: FloatArray, phi: FloatArray) -> FloatArray:
        """Expected value of the data given the signal."""

    def sample(self, signal: FloatArray, phi: FloatArray, rng: np.random.Generator) -> CountArray:
        """Draw one dataset from the observation distribution."""

    def log_prob(self, y: CountArray, signal: FloatArray, phi: FloatArray) -> FloatArray:
        """Pointwise log-density of ``y``, shaped like ``y``."""


@dataclass(frozen=True)
class GenerativeModel:
    """A system model and an observation model, composed into one generative story.

    This is what every rung of the ladder consumes, and the only object that knows
    the full parameter vector. Parameters are concatenated system-first, so for an
    SIR observed through a reporting fraction the vector is ``(beta, gamma, rho)``.
    """

    system: SystemModel
    observation: ObservationModel

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.system.param_names + self.observation.param_names

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def split(self, theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Split a full parameter vector into system and observation parts."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} parameters, got shape {theta.shape}")
        cut = len(self.system.param_names)
        return theta[:cut], theta[cut:]

    def expected(self, theta: FloatArray, t: FloatArray) -> FloatArray:
        """Expected data -- the signal pushed through the observation model's mean."""
        theta_sys, phi = self.split(theta)
        return self.observation.mean(self.system.signal(theta_sys, t), phi)

    def sample(self, theta: FloatArray, t: FloatArray, rng: np.random.Generator) -> CountArray:
        """Draw one synthetic dataset."""
        theta_sys, phi = self.split(theta)
        return self.observation.sample(self.system.signal(theta_sys, t), phi, rng)

    def log_likelihood(self, y: CountArray, theta: FloatArray, t: FloatArray) -> float:
        """Total log-likelihood of ``y``.

        Returns ``-inf`` rather than raising when the parameters are out of range, so
        that optimisers and samplers can walk into invalid regions and back out.
        """
        try:
            theta_sys, phi = self.split(theta)
            signal = self.system.signal(theta_sys, t)
            if not np.all(np.isfinite(signal)):
                return -np.inf
            total = float(np.sum(self.observation.log_prob(y, signal, phi)))
        except ValueError:
            # Out-of-range parameters, from either half of the model. Optimisers and
            # samplers routinely propose them; they need a wall, not an exception.
            return -np.inf
        return total if np.isfinite(total) else -np.inf
