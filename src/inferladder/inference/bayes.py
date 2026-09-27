"""Rung 5: the joint posterior.

Rung 4 produced intervals and said you could stop there. Two things push past it, and
neither is "better error bars":

*Prior information exists and the likelihood has nowhere to put it.* Serial-interval
studies constrain ``1/gamma``; serosurveys constrain ``rho``. A likelihood cannot hear
any of that.

*Intervals are one-dimensional and decisions are not.* Rung 3 showed ``beta`` and
``gamma`` are strongly correlated. Propagating a pair of separate intervals through a
policy question double-counts uncertainty the data already resolved. The joint posterior
carries the correlation with it.

## Why this samples without gradients

The plan called for NUTS. NUTS needs the gradient of the log-posterior, and the SIR here
is integrated by `scipy`, which is opaque to automatic differentiation. Getting gradients
would mean a second implementation of the model in a differentiable framework -- and two
implementations of the same dynamics is exactly the drift the model-agnostic design is
meant to avoid.

So the likelihood is wrapped as a black box and sampled with ``DEMetropolisZ``, which
proposes along directions learned from the chain's own history. That choice is not a
detail: a componentwise random walk *fails* on this posterior, because it can only
propose moves parallel to the axes and the probability mass lies along a diagonal ridge.
Notebook 05 shows both, and the difference is stark.

This module is not imported by ``inferladder.inference``. PyMC and ArviZ live in the
optional ``bayes`` dependency group, so importing it from the package root would make a
heavy install mandatory for everyone.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.graph import Apply, Op

from inferladder.models.base import CountArray, FloatArray, GenerativeModel


@dataclass(frozen=True)
class Priors:
    """Weakly informative priors for the SIR observed through a reporting fraction.

    Every one of these is a claim, so each is stated rather than buried:

    ``beta``, ``gamma``
        Log-normal, centred on plausible respiratory-pathogen rates and wide enough to
        span an order of magnitude. Log-normal keeps them positive without a hard edge.
    ``rho``
        Beta(2, 4): most reporting fractions are well below one half, and the shape
        rules out both 0 and 1 without pretending to know the value. This is the prior
        that does real work -- ``rho`` is the parameter the likelihood struggles with.
    ``k``
        Half-normal. Large ``k`` means Poisson-like; the prior mildly favours the
        overdispersion that surveillance data actually shows.
    """

    beta_mu: float = np.log(0.3)
    beta_sigma: float = 0.6
    gamma_mu: float = np.log(0.15)
    gamma_sigma: float = 0.6
    rho_alpha: float = 2.0
    rho_beta: float = 4.0
    k_sigma: float = 20.0


class _BlackBoxLogLikelihood(Op):
    """Wraps ``GenerativeModel.log_likelihood`` as a PyTensor operation.

    No gradient is defined, so only gradient-free samplers can use it. That is the
    honest consequence of integrating the dynamics with `scipy`.
    """

    __props__ = ()

    def __init__(self, model: GenerativeModel, y: CountArray, t: FloatArray) -> None:
        self._model = model
        self._y = y
        self._t = t

    def make_node(self, *inputs: Any) -> Apply:
        # Signature kept as wide as the base class's on purpose: PyTensor calls Ops
        # generically, so narrowing it here would be a Liskov violation for the sake
        # of a docstring.
        (theta,) = inputs
        return Apply(self, [pt.as_tensor_variable(theta)], [pt.scalar(dtype="float64")])

    def perform(self, node: Apply, inputs: Sequence[Any], outputs: Sequence[list[Any]]) -> None:
        (theta,) = inputs
        value = self._model.log_likelihood(self._y, np.asarray(theta, dtype=np.float64), self._t)
        outputs[0][0] = np.asarray(value, dtype="float64")


def build_model(
    model: GenerativeModel,
    y: CountArray,
    t: FloatArray,
    priors: Priors | None = None,
) -> pm.Model:
    """A PyMC model over ``(beta, gamma, rho, k)`` with the SIR likelihood as a black box."""
    if model.param_names != ("beta", "gamma", "rho", "k"):
        raise ValueError(
            "this prior set is written for (beta, gamma, rho, k); "
            f"got {model.param_names}. Add a Priors variant for the new model."
        )
    priors = priors or Priors()
    log_likelihood = _BlackBoxLogLikelihood(model, y, t)

    with pm.Model() as pymc_model:
        beta = pm.LogNormal("beta", mu=priors.beta_mu, sigma=priors.beta_sigma)
        gamma = pm.LogNormal("gamma", mu=priors.gamma_mu, sigma=priors.gamma_sigma)
        rho = pm.Beta("rho", alpha=priors.rho_alpha, beta=priors.rho_beta)
        k = pm.HalfNormal("k", sigma=priors.k_sigma)
        pm.Deterministic("R0", beta / gamma)
        pm.Potential("likelihood", log_likelihood(pt.stack([beta, gamma, rho, k])))

    return pymc_model


def sample_posterior(
    model: GenerativeModel,
    y: CountArray,
    t: FloatArray,
    *,
    priors: Priors | None = None,
    draws: int = 2_000,
    tune: int = 2_000,
    chains: int = 4,
    cores: int | None = None,
    seed: int = 0,
    sampler: str = "demetropolisz",
    progressbar: bool = False,
) -> xr.DataTree:
    """Sample the posterior and return the draws as an ``xarray.DataTree``.

    That is PyMC 6's return type: ArviZ 1.x retired ``InferenceData`` in favour of
    xarray's ``DataTree``, and the old name survives only as a deprecation shim.

    ``sampler`` selects between ``"demetropolisz"`` -- the default, which learns
    proposal directions from the chain history and handles the correlated ridge -- and
    ``"metropolis"``, a componentwise random walk kept available *because* it fails
    here. Being able to reproduce the failure is worth more than hiding it.

    ``tune`` is deliberately as large as ``draws``: a gradient-free sampler needs the
    warm-up to learn the shape of the ridge, and cutting it is the fastest way to get
    confident nonsense.
    """
    steps = {"demetropolisz": pm.DEMetropolisZ, "metropolis": pm.Metropolis}
    if sampler not in steps:
        raise ValueError(f"unknown sampler {sampler!r}; choose from {sorted(steps)}")

    with build_model(model, y, t, priors):
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains if cores is None else cores,
            step=steps[sampler](),
            random_seed=seed,
            progressbar=progressbar,
        )


@dataclass(frozen=True)
class ConvergenceReport:
    """Whether a sample can be trusted, separate from what it says."""

    max_r_hat: float
    min_ess_bulk: float
    n_divergences: int
    per_parameter: dict[str, tuple[float, float]] = field(repr=False, default_factory=dict)

    @property
    def converged(self) -> bool:
        """The usual thresholds: R-hat below 1.01 and at least 400 effective draws.

        Deliberately strict. A sample that fails this is not "roughly right" -- on a
        correlated posterior a stuck chain reports a *narrower* interval than the truth
        deserves, so the failure mode is overconfidence, not noise.
        """
        return self.max_r_hat < 1.01 and self.min_ess_bulk >= 400


def check_convergence(idata: xr.DataTree, var_names: list[str] | None = None) -> ConvergenceReport:
    """Summarise R-hat and effective sample size across parameters."""
    summary = az.summary(idata, var_names=var_names, round_to=None)
    per_parameter = {
        str(name): (float(row["r_hat"]), float(row["ess_bulk"])) for name, row in summary.iterrows()
    }
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum())

    return ConvergenceReport(
        max_r_hat=float(summary["r_hat"].max()),
        min_ess_bulk=float(summary["ess_bulk"].min()),
        n_divergences=divergences,
        per_parameter=per_parameter,
    )


def posterior_samples(idata: xr.DataTree, param_names: tuple[str, ...]) -> FloatArray:
    """Flatten the posterior into a ``(n_draws, n_params)`` array.

    The bridge between rungs: these draws and rung 4's bootstrap replicates have the
    same shape, so one code path compares them -- and so would any other sampler's.
    """
    posterior = idata.posterior
    columns = [posterior[name].to_numpy().reshape(-1) for name in param_names]
    return np.column_stack(columns)
