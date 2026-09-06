"""Rung 3: do the data actually distinguish the parameters?

Rungs 1 and 2 return a point. This module asks whether that point means anything.

The machinery is all one idea seen three ways. Take the map from parameters to
predictions and look at how it stretches:

``sensitivity_spectrum``
    Singular values of the Jacobian of the predictions with respect to the parameters.
    A large spread means some *combinations* of parameters change the prediction a lot
    and others barely change it at all. The near-null directions are the ones the data
    cannot see -- the "sloppy" directions. This says nothing about noise; it is a
    property of the model and the sampling times alone.

``observed_fisher``
    The curvature of the log-likelihood at the estimate, which does involve the noise.
    Its inverse is the Laplace approximation to the covariance. A near-singular Fisher
    matrix and a near-null singular direction are the same finding.

``profile_likelihood``
    The honest version. For each value of one parameter, re-optimise all the others and
    record the best achievable fit. This is what distinguishes a parameter that is
    genuinely pinned down from one that merely looks pinned down when its partners are
    held fixed -- and unlike the other two it makes no quadratic assumption, so it
    still works when the likelihood is skewed.

Everything works in log-parameter space. These are positive rates spanning different
scales, and a sensitivity expressed in absolute units would just report which parameter
happens to be numerically largest.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from inferladder.inference.nls_mle import Objective, fit
from inferladder.models.base import FloatArray


@dataclass(frozen=True)
class SensitivitySpectrum:
    """Singular value decomposition of the parameter-to-prediction Jacobian."""

    singular_values: FloatArray
    directions: FloatArray = field(repr=False)
    param_names: tuple[str, ...]

    @property
    def condition_number(self) -> float:
        """Ratio of the largest to smallest singular value.

        The usual summary of sloppiness. Large means at least one parameter combination
        is invisible to the data at the precision the rest are known to.
        """
        smallest = self.singular_values[-1]
        return float(np.inf if smallest <= 0 else self.singular_values[0] / smallest)

    def stiffest(self) -> dict[str, float]:
        """The best-determined parameter combination, as loadings per parameter."""
        return dict(zip(self.param_names, map(float, self.directions[0]), strict=True))

    def sloppiest(self) -> dict[str, float]:
        """The worst-determined combination -- the direction along the valley floor."""
        return dict(zip(self.param_names, map(float, self.directions[-1]), strict=True))


def log_jacobian(
    predict: Callable[[FloatArray], FloatArray],
    theta: FloatArray,
    *,
    rel_step: float = 1e-5,
) -> FloatArray:
    """Jacobian of ``log(predict)`` with respect to ``log(theta)``, by central differences.

    Both sides in logs, so an entry reads as a relative change in the prediction per
    relative change in the parameter -- an elasticity, comparable across parameters
    that live on different scales.
    """
    theta = np.asarray(theta, dtype=np.float64)
    columns = []
    for index in range(theta.size):
        step = rel_step * max(abs(np.log(theta[index])), 1.0)
        up, down = theta.astype(np.float64).copy(), theta.astype(np.float64).copy()
        up[index] = np.exp(np.log(theta[index]) + step)
        down[index] = np.exp(np.log(theta[index]) - step)
        forward = np.log(np.maximum(predict(up), 1e-300))
        backward = np.log(np.maximum(predict(down), 1e-300))
        columns.append((forward - backward) / (2 * step))
    return np.column_stack(columns)


def sensitivity_spectrum(
    predict: Callable[[FloatArray], FloatArray],
    theta: FloatArray,
    param_names: tuple[str, ...],
    *,
    rel_step: float = 1e-5,
) -> SensitivitySpectrum:
    """Singular values and directions of the log-log Jacobian at ``theta``."""
    jacobian = log_jacobian(predict, theta, rel_step=rel_step)
    n_predictions, n_parameters = jacobian.shape
    if n_predictions < n_parameters:
        # With fewer predictions than parameters the decomposition returns fewer
        # directions than there are parameters, and `sloppiest` would name a direction
        # that is merely missing rather than one the data cannot see. Refusing is
        # better than reporting a spectrum that is guaranteed to be incomplete.
        raise ValueError(
            f"{n_predictions} predictions cannot resolve {n_parameters} parameters; "
            "the spectrum would be structurally rank-deficient"
        )
    _, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    return SensitivitySpectrum(
        singular_values=singular_values,
        directions=right,
        param_names=param_names,
    )


def observed_fisher(
    objective: Objective,
    theta: FloatArray,
    *,
    rel_step: float = 1e-4,
) -> FloatArray:
    """Hessian of a negative log-likelihood at ``theta``, in log-parameter space.

    Central differences on an objective that itself solves an ODE, so the result is
    only as smooth as the solver tolerances. Loose tolerances show up here as a matrix
    that is not quite symmetric; it is symmetrised before being returned, which hides
    the symptom but not the cause.
    """
    theta = np.asarray(theta, dtype=np.float64)
    u = np.log(theta)
    n = u.size
    step = rel_step * np.maximum(np.abs(u), 1.0)

    def at(point: FloatArray) -> float:
        return objective(np.exp(point))

    center = at(u)
    hessian = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                up, down = u.copy(), u.copy()
                up[i] += step[i]
                down[i] -= step[i]
                hessian[i, i] = (at(up) - 2 * center + at(down)) / step[i] ** 2
            else:
                shifted = {}
                for si in (+1, -1):
                    for sj in (+1, -1):
                        point = u.copy()
                        point[i] += si * step[i]
                        point[j] += sj * step[j]
                        shifted[(si, sj)] = at(point)
                value = (
                    shifted[(+1, +1)] - shifted[(+1, -1)] - shifted[(-1, +1)] + shifted[(-1, -1)]
                ) / (4 * step[i] * step[j])
                hessian[i, j] = hessian[j, i] = value
    return 0.5 * (hessian + hessian.T)


@dataclass(frozen=True)
class Profile:
    """A profile likelihood for one parameter."""

    name: str
    values: FloatArray
    objective: FloatArray
    minimum: float

    @property
    def delta(self) -> FloatArray:
        """Rise above the best fit -- the quantity a confidence threshold cuts."""
        return self.objective - self.minimum


def profile_likelihood(
    objective: Objective,
    theta_hat: FloatArray,
    index: int,
    grid: Sequence[float],
    param_names: tuple[str, ...],
    *,
    method: str = "Nelder-Mead",
    max_evaluations: int = 2_000,
) -> Profile:
    """Re-optimise every other parameter at each fixed value of parameter ``index``.

    This is the expensive diagnostic -- one full optimisation per grid point -- and the
    only one that answers the question honestly. Slicing instead (holding the others at
    their fitted values) makes every parameter look better determined than it is.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    free = [k for k in range(theta_hat.size) if k != index]
    free_names = tuple(param_names[k] for k in free)

    if not free:
        raise ValueError("profiling requires at least one other parameter to re-optimise")

    results = []
    for fixed_value in grid:

        def restricted(reduced: FloatArray, _fixed: float = fixed_value) -> float:
            full = theta_hat.copy()
            full[index] = _fixed
            full[free] = reduced
            return objective(full)

        estimate = fit(
            restricted,
            theta_hat[free],
            free_names,
            method=method,
            max_evaluations=max_evaluations,
        )
        results.append(estimate.objective_value)

    objective_values = np.asarray(results, dtype=np.float64)
    return Profile(
        name=param_names[index],
        values=np.asarray(grid, dtype=np.float64),
        objective=objective_values,
        minimum=float(objective_values.min()),
    )


def profile_interval(profile: Profile, level: float = 0.95) -> tuple[float, float]:
    """Cut a profile at the chi-squared threshold for one degree of freedom.

    Returns an infinite bound on either side where the profile never rises above the
    threshold within the grid: the data do not bound the parameter there, and reporting
    the edge of the grid as a limit would be an artefact of where the grid was drawn.
    """
    threshold = 0.5 * stats.chi2.ppf(level, df=1)
    inside = profile.delta <= threshold
    if not inside.any():
        raise ValueError("no grid point lies inside the interval; the grid misses the optimum")

    lower = -np.inf if inside[0] else float(profile.values[inside].min())
    upper = np.inf if inside[-1] else float(profile.values[inside].max())
    return (lower, upper)
