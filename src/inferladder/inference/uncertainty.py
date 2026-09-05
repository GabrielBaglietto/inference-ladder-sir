"""Rung 4: intervals, and the place where you are allowed to stop.

This is the rung the plan singles out. If the question is "fit this model and say how
confident you are", the answer is here, and climbing further buys nothing. The reason
to keep going is not better error bars -- it is wanting a *joint* distribution to push
through a decision, which is rung 5.

Three ways to get an interval, in increasing order of cost and honesty:

``laplace_intervals``
    Invert the curvature at the optimum and read off standard errors. One Hessian, and
    then it is free. It assumes the likelihood is a Gaussian bump in log-parameters,
    which is exactly the assumption that fails when a parameter is weakly identified --
    the case where you most need the interval. Cheap, and it fails quietly.

``diagnostics.profile_interval``
    Cut the profile likelihood at a chi-squared threshold. No quadratic assumption, so
    it handles skew and can return an unbounded side when a parameter is genuinely not
    pinned down. Costs one optimisation per grid point. It lives next to the profile
    machinery in ``diagnostics``, because cutting a profile is an operation on a profile.

``parametric_bootstrap``
    Simulate new datasets from the fitted model, refit each, look at the spread. Makes
    no assumption about the shape of the likelihood, and it also exercises the whole
    estimation pipeline, so it catches an optimiser that only sometimes converges.
    Costs one full fit per replicate, and it is the honest way to discover that your
    estimator is less stable than a single run suggested.

The three agreeing is real evidence. Where they disagree, the cheap one is wrong.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from inferladder.inference.nls_mle import PointEstimate, fit, negative_log_likelihood
from inferladder.models.base import FloatArray, GenerativeModel

Interval = tuple[float, float]


@dataclass(frozen=True)
class BootstrapResult:
    """Refits of the model on data simulated from the fitted parameters."""

    samples: FloatArray
    param_names: tuple[str, ...]
    n_failed: int

    def interval(self, name: str, level: float = 0.95) -> Interval:
        """Percentile interval for one parameter."""
        column = self.samples[:, self.param_names.index(name)]
        tail = 100 * (1 - level) / 2
        return (float(np.percentile(column, tail)), float(np.percentile(column, 100 - tail)))

    def derived(self, func: Callable[[FloatArray], float]) -> FloatArray:
        """Apply a function of the parameter vector to every replicate.

        The reason bootstrap is worth its cost: quantities like ``R0 = beta / gamma``
        get an interval for free, correlations included, with no delta-method algebra.
        """
        return np.array([float(func(row)) for row in self.samples])


def laplace_covariance(hessian: FloatArray) -> FloatArray:
    """Covariance of the log-parameters from the Hessian of the negative log-likelihood.

    Raises if the Hessian is not positive definite. That failure is informative and
    should not be papered over with a pseudo-inverse: it means the optimum is not a
    well-defined peak in at least one direction, which is a finding about the model,
    not a numerical nuisance.
    """
    hessian = np.asarray(hessian, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues.min() <= 0:
        raise np.linalg.LinAlgError(
            f"Hessian is not positive definite (smallest eigenvalue {eigenvalues.min():.3e}); "
            "the likelihood has no curvature in at least one direction"
        )
    return np.linalg.inv(hessian)


def laplace_intervals(
    theta_hat: FloatArray,
    hessian: FloatArray,
    param_names: tuple[str, ...],
    level: float = 0.95,
) -> dict[str, Interval]:
    """Wald intervals computed in log space and mapped back to the parameters.

    Mapping back by exponentiating gives asymmetric intervals on the original scale,
    which is both more accurate and a better picture of a positive quantity than a
    symmetric interval that can dip below zero.
    """
    covariance = laplace_covariance(hessian)
    critical = stats.norm.ppf(0.5 * (1 + level))
    log_theta = np.log(np.asarray(theta_hat, dtype=np.float64))
    errors = np.sqrt(np.diag(covariance))
    return {
        name: (
            float(np.exp(log_theta[i] - critical * errors[i])),
            float(np.exp(log_theta[i] + critical * errors[i])),
        )
        for i, name in enumerate(param_names)
    }


def parametric_bootstrap(
    model: GenerativeModel,
    theta_hat: FloatArray,
    t: FloatArray,
    *,
    n_replicates: int = 200,
    seed: int = 0,
    method: str = "Nelder-Mead",
    max_evaluations: int = 2_000,
) -> BootstrapResult:
    """Simulate from the fit, refit each replicate, and collect the estimates.

    Replicates whose optimisation fails are dropped and counted rather than silently
    included. A high failure count is itself the result: it says the estimator is not
    reliable on data like the data in hand.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    rng = np.random.default_rng(seed)

    estimates: list[FloatArray] = []
    failures = 0
    for _ in range(n_replicates):
        replicate = model.sample(theta_hat, t, rng)
        estimate: PointEstimate = fit(
            negative_log_likelihood(model, replicate, t),
            theta_hat,
            model.param_names,
            method=method,
            max_evaluations=max_evaluations,
        )
        if estimate.success and np.all(np.isfinite(estimate.theta)):
            estimates.append(estimate.theta)
        else:
            failures += 1

    if not estimates:
        raise RuntimeError(f"every one of {n_replicates} bootstrap refits failed")

    return BootstrapResult(
        samples=np.vstack(estimates),
        param_names=model.param_names,
        n_failed=failures,
    )


def normal_approximation_intervals(samples: Sequence[float], level: float = 0.95) -> Interval:
    """Mean +/- z * sd of a sample, for comparing against a percentile interval.

    Where this and the percentile interval disagree, the sampling distribution is
    skewed and the symmetric summary is the one to distrust.
    """
    values = np.asarray(samples, dtype=np.float64)
    critical = stats.norm.ppf(0.5 * (1 + level))
    half_width = critical * values.std(ddof=1)
    return (float(values.mean() - half_width), float(values.mean() + half_width))


__all__ = [
    "BootstrapResult",
    "Interval",
    "laplace_covariance",
    "laplace_intervals",
    "normal_approximation_intervals",
    "parametric_bootstrap",
]
