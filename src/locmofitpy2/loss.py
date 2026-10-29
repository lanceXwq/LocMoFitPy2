"""The likelihood of LocMoFit.jl's derivations/derivation.qmd: each localization's density is a mixture of the structure (the mean of Gaussians over each component's model points, the components weighted) and a uniform background."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def log_likelihood(
    locs, loc_prec, points, ranges, sigma, logw, a, b, background
):
    """Σₙ log(eᵃ pₙ + eᵇ) over the localizations (the rows of the (N, D) `locs`, with precisions `loc_prec`), or Σₙ log pₙ without a background.

    pₙ is the structure's density at localization n: for each component k, whose model points are the rows `ranges[k]` of `points` (of which the first D columns are used), the mean of the Gaussians around its points with per-axis variances `loc_prec² + sigma²`; the components weighted by `exp(logw)`. In a fit, `a` is the log signal fraction and `b` the log of the background's density times its fraction. The squared distances are formed from differences (not expanded), so the result keeps its precision far from the origin, also in float32.
    """
    D = locs.shape[1]
    v = loc_prec**2 + sigma**2  # (N, D)
    c = (
        -jnp.sum(jnp.log(2 * math.pi * v), axis=1) / 2
    )  # the Gaussians' normalization
    d = locs[:, None, :] - points[None, :, :D]  # (N, M, D)
    q = jnp.sum(d**2 / (2 * v[:, None, :]), axis=2)  # (N, M)
    terms = [
        logw[k]
        - math.log(r.stop - r.start)
        + logsumexp(-q[:, r.start : r.stop], axis=1)
        for k, r in enumerate(ranges)
    ]
    S = terms[0] if len(terms) == 1 else logsumexp(jnp.stack(terms), axis=0)
    if not background:
        return jnp.sum(c + S)
    return jnp.sum(jnp.logaddexp(a + c + S, b))


def make_loss(problem):
    """The objective of a fit, as a function of the free parameters `u`: minus the log posterior of the parameters, per localization, plus the model's penalty (`problem` from `fit_problem`)."""
    model, pmap, unit = (
        problem.model,
        problem.parameter_map,
        problem.unit_model,
    )
    locs, prec = problem.locs, problem.loc_prec
    N = locs.shape[0]

    def loss(u):
        theta = pmap.full_parameters(u)
        points = model.model_points(unit, theta)
        lw = jnp.stack(model.log_weights(theta))
        logw = lw - logsumexp(lw)
        if not problem.background:
            ll = log_likelihood(
                locs,
                prec,
                points,
                problem.ranges,
                theta["sigma"],
                logw,
                0.0,
                0.0,
                False,
            )
            return -ll / N + model.penalty(theta)
        lam = theta["logit_prob"]
        log_pi, log_1mpi = jax.nn.log_sigmoid(lam), jax.nn.log_sigmoid(-lam)
        ll = log_likelihood(
            locs,
            prec,
            points,
            problem.ranges,
            theta["sigma"],
            logw,
            log_pi,
            log_1mpi - problem.log_volume,
            True,
        )
        if problem.signal_prior is None:
            prior = jnp.zeros_like(log_pi)
        else:
            a, b = problem.signal_prior
            prior = (a - 1) * log_pi + (b - 1) * log_1mpi
        return -(ll + prior) / N + model.penalty(theta)

    return loss
