import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from .utils import Data


def pairwise_sqdist(X: Array, Y: Array, invSigma: Array) -> Array:
    """
    Weighted pairwise squared distances.

    Args:
        X:        (N, 3) points
        Y:        (M, 3) points
        invSigma: (N, 3) per-X-point diagonal weights (precisions) for each dimension

    Returns:
        D2: (M, N) where
            D2[m, n] = sum_d invSigma[n, d] * (Y[m, d] - X[n, d])**2
    """
    term1 = (Y * Y) @ invSigma.T
    term2 = jnp.sum(invSigma * (X * X), axis=1)
    term3 = Y @ (invSigma * X).T

    return term1 + term2[None, :] - jnp.array(2.0, dtype=X.dtype) * term3


pairwise_sqdist_jit = jax.jit(pairwise_sqdist)


def negative_log_likelihood(
    x_data: Array, x_model: Array, half_tau: Array, log_const: Array
) -> Array:
    """
    Args:
        datapoints: (N, 3)
        modelpoints: (M, 3)
        half_tau: (N, 3)  per-datapoint diagonal weights (precisions/2), used in pairwise_sqdist
        log_const: scalar or (N,) or (1,N), broadcastable to (M, N) as log_const - Δx²

    Returns:
        nll: scalar (0-dim) JAX array
    """
    sqdist = pairwise_sqdist_jit(x_data, x_model, half_tau)
    per_pair_lls = log_const - sqdist
    lse = logsumexp(per_pair_lls, axis=0)
    return -jnp.sum(lse)


def loss(static, data: Data):
    def nll_loss(trainable):
        model = eqx.combine(trainable, static)
        positions = model()
        return negative_log_likelihood(
            data.locs, positions, data.half_precisions, data.log_consts
        )

    return nll_loss
