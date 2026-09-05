"""Rung 2: when the optimiser is the problem.

Rung 1 assumes the minimiser finds the minimum. It often does not. Two failure modes
show up on this model, and they need different responses:

*Local minima and flat valleys.* A single local search from a single starting point
returns whatever basin it happened to start in, with `success: True` either way.
``multistart`` runs the same search from many starts and shows the spread. If every
start converges to the same place, that is evidence; if they scatter, the point
estimate from rung 1 was never a fact.

*A cost surface that is not shaped like the textbook picture.* ``cost_surface``
evaluates the objective on a grid so it can be looked at. On an SIR the valley is
long, curved and nearly flat along one direction -- which is the *reason* the optimiser
struggles, and is exactly what rung 3 goes on to quantify.

A note on cost: every objective evaluation solves an ODE. A 60x60 surface is 3600
solves. The grids here are meant to be coarse enough to look at, not to publish.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from inferladder.inference.nls_mle import Objective, PointEstimate, fit
from inferladder.models.base import FloatArray


@dataclass(frozen=True)
class CostSurface:
    """Objective values on a 2-d grid, ready to contour."""

    x: FloatArray
    y: FloatArray
    values: FloatArray
    x_name: str
    y_name: str

    @property
    def minimum(self) -> tuple[float, float]:
        """Grid coordinates of the lowest evaluated point."""
        flat = int(np.nanargmin(self.values))
        row, col = np.unravel_index(flat, self.values.shape)
        return float(self.x[col]), float(self.y[row])


def multistart(
    objective: Objective,
    starts: Sequence[FloatArray],
    param_names: tuple[str, ...],
    *,
    method: str = "Nelder-Mead",
    max_evaluations: int = 5_000,
) -> tuple[PointEstimate, list[PointEstimate]]:
    """Run a local search from every start; return the best one and all of them.

    Returning all of them is the point. The best fit alone looks just as convincing
    whether the others agreed with it or scattered across the parameter space.
    """
    if len(starts) == 0:
        raise ValueError("need at least one starting point")

    runs = [
        fit(objective, start, param_names, method=method, max_evaluations=max_evaluations)
        for start in starts
    ]
    best = min(runs, key=lambda run: run.objective_value)
    return best, runs


def random_starts(
    center: FloatArray,
    n: int,
    *,
    spread: float = 0.5,
    seed: int = 0,
) -> list[FloatArray]:
    """Draw ``n`` starting points log-uniformly around ``center``.

    ``spread`` is in log units: 0.5 means roughly a factor of 1.6 either way. Scattering
    multiplicatively rather than additively keeps the starts sensible for parameters
    that differ by orders of magnitude.
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=np.float64)
    offsets = rng.uniform(-spread, spread, size=(n, center.size))
    result: list[FloatArray] = list(np.exp(np.log(center) + offsets))
    return result


def cost_surface(
    objective: Objective,
    theta: FloatArray,
    indices: tuple[int, int],
    param_names: tuple[str, ...],
    *,
    factor: float = 1.5,
    resolution: int = 40,
) -> CostSurface:
    """Evaluate ``objective`` over a 2-d slice through ``theta``.

    The two named parameters vary log-uniformly within ``factor`` of their values; the
    rest are held fixed. Holding them fixed is a real limitation -- a slice is not a
    profile, and a valley that looks narrow here can be wide once the other parameters
    are free to compensate. Rung 3 draws that distinction properly.
    """
    theta = np.asarray(theta, dtype=np.float64)
    i, j = indices
    if i == j:
        raise ValueError("need two different parameters to make a surface")
    if factor <= 1.0:
        raise ValueError("factor must exceed 1")

    x = np.exp(np.linspace(np.log(theta[i] / factor), np.log(theta[i] * factor), resolution))
    y = np.exp(np.linspace(np.log(theta[j] / factor), np.log(theta[j] * factor), resolution))

    values = np.empty((resolution, resolution), dtype=np.float64)
    for row, y_value in enumerate(y):
        for col, x_value in enumerate(x):
            candidate = theta.copy()
            candidate[i] = x_value
            candidate[j] = y_value
            values[row, col] = objective(candidate)

    return CostSurface(x=x, y=y, values=values, x_name=param_names[i], y_name=param_names[j])
