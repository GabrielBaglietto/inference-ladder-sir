"""Rungs 1 and 2: fitting the whole curve.

Rung 0 used one straight line and got one number. To get ``beta`` and ``gamma``
separately the whole curve has to be fitted, which means solving the ODE inside an
optimiser and choosing what "best fit" means.

Two objectives live here, and the gap between them is the point of rung 1:

``sum_of_squares``
    Ordinary least squares on the counts. Equivalent to maximum likelihood *if* the
    noise is Gaussian with constant variance -- an assumption epidemic counts violate
    badly, since the variance grows with the mean. Fitting it anyway lets the peak of
    the curve dominate and gives the early data, where the growth rate lives, almost
    no weight.

``negative_log_likelihood``
    The likelihood of the observation model actually being used. Poisson or negative
    binomial weight each point by how noisy that point is expected to be.

Both are minimised in log-parameter space. Every parameter here is a positive rate or
a fraction, and an unconstrained optimiser handed the raw values spends its time
proposing negative rates and being rejected.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from inferladder.models.base import CountArray, FloatArray, GenerativeModel

Objective = Callable[[FloatArray], float]


@dataclass(frozen=True)
class PointEstimate:
    """The result of one optimisation run."""

    theta: FloatArray
    param_names: tuple[str, ...]
    objective_value: float
    success: bool
    n_evaluations: int
    method: str
    message: str = field(default="", repr=False)

    @property
    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.param_names, map(float, self.theta), strict=True))

    def __getitem__(self, name: str) -> float:
        return self.as_dict[name]


def negative_log_likelihood(model: GenerativeModel, y: CountArray, t: FloatArray) -> Objective:
    """Build ``theta -> -log L(theta)`` for the given data."""

    def objective(theta: FloatArray) -> float:
        return -model.log_likelihood(y, theta, t)

    return objective


def sum_of_squares(model: GenerativeModel, y: CountArray, t: FloatArray) -> Objective:
    """Build ``theta -> sum (y - E[y|theta])^2`` for the given data.

    Kept deliberately naive -- unweighted, on the raw counts -- because that is the
    estimator rung 1 compares against, and softening it here would hide the comparison.
    """

    def objective(theta: FloatArray) -> float:
        try:
            residuals = np.asarray(y, dtype=np.float64) - model.expected(theta, t)
        except ValueError:
            return np.inf
        total = float(np.sum(residuals**2))
        return total if np.isfinite(total) else np.inf

    return objective


def fit(
    objective: Objective,
    theta0: FloatArray,
    param_names: tuple[str, ...],
    *,
    method: str = "Nelder-Mead",
    max_evaluations: int = 5_000,
) -> PointEstimate:
    """Minimise ``objective`` starting from ``theta0``, working in log-parameter space."""
    theta0 = np.asarray(theta0, dtype=np.float64)
    if np.any(theta0 <= 0):
        raise ValueError("all starting parameters must be positive to optimise in log space")
    if len(param_names) != theta0.size:
        raise ValueError(f"got {theta0.size} parameters and {len(param_names)} names")

    def in_log_space(u: FloatArray) -> float:
        return objective(np.exp(u))

    result = minimize(
        in_log_space,
        np.log(theta0),
        method=method,
        options={"maxfev": max_evaluations} if method == "Nelder-Mead" else None,
    )
    return PointEstimate(
        theta=np.exp(result.x),
        param_names=param_names,
        objective_value=float(result.fun),
        success=bool(result.success),
        n_evaluations=int(getattr(result, "nfev", 0)),
        method=method,
        message=str(result.message),
    )
