"""Rung 0: the growth rate from a straight line.

Early in an epidemic, while susceptibles are still essentially the whole population,
incidence grows as ``exp(r t)`` with ``r = beta - gamma``. Taking logs makes that a
straight line, and a straight line can be fitted in closed form. This is the honest
linear entry point: no solver, no starting values, no convergence to worry about.

It buys one number, and it is worth being precise about which. ``r`` is a *combination*
of the two rates. Every pair ``(beta, gamma)`` with the same difference fits the early
data equally well, which is the first appearance of the identifiability problem that
rung 3 makes explicit. Rung 0 cannot separate them, and pretending otherwise is where
epidemic estimates go wrong.

The window matters as much as the fit. Too early and the counts are small and dominated
by noise; too late and susceptible depletion has bent the curve, biasing ``r`` downward.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from inferladder.models.base import CountArray, FloatArray


@dataclass(frozen=True)
class LinearFit:
    """A least-squares line through log-counts, with the standard error of its slope."""

    slope: float
    intercept: float
    slope_stderr: float
    r_squared: float
    n_points: int
    window: tuple[float, float]

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Confidence interval for the slope, from the t distribution.

        Uses ``n - 2`` degrees of freedom: the line spends two of them. With the short
        windows typical of an epidemic's exponential phase that correction is not a
        formality -- it visibly widens the interval.
        """
        if not 0 < level < 1:
            raise ValueError("level must lie in (0, 1)")
        if self.n_points <= 2:
            return (-np.inf, np.inf)
        critical = stats.t.ppf(0.5 * (1 + level), df=self.n_points - 2)
        half_width = critical * self.slope_stderr
        return (self.slope - half_width, self.slope + half_width)

    def doubling_time(self) -> float:
        """Days for cases to double at this growth rate; ``inf`` if not growing."""
        return float(np.log(2.0) / self.slope) if self.slope > 0 else np.inf


def exponential_growth_rate(
    t: FloatArray,
    y: CountArray,
    window: tuple[float, float],
) -> LinearFit:
    """Fit ``log(y) ~ a + r t`` over ``window`` and return the slope ``r``.

    Zero counts are dropped rather than shifted. Adding a pseudo-count to keep them is
    a common reflex, but it distorts precisely the smallest observations, which sit at
    the start of the window where the fit has the most leverage.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y)
    if t.shape != y.shape:
        raise ValueError(f"t and y must have the same shape, got {t.shape} and {y.shape}")

    start, end = window
    if not start < end:
        raise ValueError(f"window must be increasing, got {window}")

    inside = (t >= start) & (t <= end) & (y > 0)
    if inside.sum() < 3:
        raise ValueError(
            f"window {window} leaves {int(inside.sum())} usable points; need at least 3"
        )

    result = stats.linregress(t[inside], np.log(y[inside]))
    return LinearFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        slope_stderr=float(result.stderr),
        r_squared=float(result.rvalue**2),
        n_points=int(inside.sum()),
        window=window,
    )
