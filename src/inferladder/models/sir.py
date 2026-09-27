"""Deterministic SIR model.

The compartmental core every rung runs on. Three states plus a cumulative-incidence
accumulator ``C``, which exists purely so that incidence can be read off exactly as
``C(t_i) - C(t_i-1)`` instead of being approximated from ``I``. That distinction
matters: ``I`` is prevalence (how many are infectious now), while case counts are
incidence (how many became infectious since the last report). Fitting one to the
other is a bias, not a rounding error.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.integrate import solve_ivp

from inferladder.models.base import FloatArray, uniform_step


@dataclass(frozen=True)
class SIR:
    """SIR dynamics on a closed population.

    Parameters
    ----------
    population:
        Total population ``N``, held fixed and known. Treating it as unknown makes
        the model badly non-identifiable from a single incidence curve.
    i0:
        Infectious individuals at the start of the first reporting period.
    rtol, atol:
        Solver tolerances. Tight by default: the likelihood surface is compared
        across nearby parameters in later rungs, and loose tolerances there show up
        as spurious roughness that looks like a real optimisation difficulty.
    """

    population: float = 100_000.0
    i0: float = 10.0
    rtol: float = 1e-9
    atol: float = 1e-9

    param_names: ClassVar[tuple[str, ...]] = ("beta", "gamma")
    state_names: ClassVar[tuple[str, ...]] = ("S", "I", "R")

    def __post_init__(self) -> None:
        if self.population <= 0:
            raise ValueError("population must be positive")
        if not 0 < self.i0 < self.population:
            raise ValueError("i0 must lie strictly between 0 and the population")

    def initial_state(self) -> FloatArray:
        """State at the start of the first reporting period: ``(S, I, R, C)``."""
        return np.array([self.population - self.i0, self.i0, 0.0, 0.0], dtype=np.float64)

    def basic_reproduction_number(self, theta: FloatArray) -> float:
        """``R0 = beta / gamma``.

        Worth naming explicitly: later rungs show that the data pin this combination
        down far more tightly than they pin down ``beta`` and ``gamma`` separately.
        """
        beta, gamma = self._unpack(theta)
        return beta / gamma

    def simulate(self, theta: FloatArray, t: FloatArray) -> FloatArray:
        """Latent ``(S, I, R)`` at each time in ``t``."""
        return self._integrate(theta, t)[:, :3]

    def signal(self, theta: FloatArray, t: FloatArray) -> FloatArray:
        """Expected incidence in the reporting period ending at each time in ``t``."""
        cumulative = self._integrate(theta, t, include_start=True)[:, 3]
        return np.diff(cumulative)

    def _unpack(self, theta: FloatArray) -> tuple[float, float]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (2,):
            raise ValueError(f"expected (beta, gamma), got shape {theta.shape}")
        beta, gamma = float(theta[0]), float(theta[1])
        if beta <= 0 or gamma <= 0:
            raise ValueError("beta and gamma must be positive")
        return beta, gamma

    def _rhs(self, _t: float, x: FloatArray, beta: float, gamma: float) -> FloatArray:
        s, i = x[0], x[1]
        infection = beta * s * i / self.population
        recovery = gamma * i
        return np.array([-infection, infection - recovery, recovery, infection])

    def _integrate(
        self, theta: FloatArray, t: FloatArray, *, include_start: bool = False
    ) -> FloatArray:
        """Integrate from one period before ``t[0]`` so period 0 is a full period."""
        beta, gamma = self._unpack(theta)
        t = np.asarray(t, dtype=np.float64)
        dt = uniform_step(t)
        grid = np.concatenate(([t[0] - dt], t))

        solution = solve_ivp(
            self._rhs,
            (grid[0], grid[-1]),
            self.initial_state(),
            t_eval=grid,
            args=(beta, gamma),
            method="LSODA",
            rtol=self.rtol,
            atol=self.atol,
        )
        if not solution.success:
            raise RuntimeError(f"SIR integration failed: {solution.message}")

        states: FloatArray = solution.y.T
        return states if include_start else states[1:]
