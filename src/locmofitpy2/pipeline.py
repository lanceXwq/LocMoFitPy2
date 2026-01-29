from __future__ import annotations

from typing import Any, Dict, Tuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .loss import loss
from .models.registry import get_model_cls
from .optim import fit_lbfgs
from .utils import Data, partition_with_freeze


def run_locmofit(
    model_name: str,
    locs: np.ndarray | jnp.ndarray,
    stddev: np.ndarray | jnp.ndarray,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    c: float = 0.02,  # 50 nm
    alpha: float = np.pi / 2,
    theta: float = 0.0,
    phi: float = 0.0,
    spacing: float = 3.0,  # 3 nm
    freeze: Tuple[str, ...] = (),
    max_iter: int = 200,
    tol: float = 1e-6,
    dtype=jnp.float32,
) -> Dict[str, Any]:
    """
    High-level fitting pipeline.

    Returns a Dict containing:
      - final loss
      - number of iterations
      - final gradient norm
      - model points as a NumPy array
      - optimized parameters as a Dict
    """
    locs_j = jnp.asarray(locs, dtype=dtype)
    stddev_j = jnp.asarray(stddev, dtype=dtype)
    data = Data.from_arrays(locs_j, stddev_j)

    ModelCls = get_model_cls(model_name)
    model0 = ModelCls.init(
        x=x,
        y=y,
        z=z,
        c=c,
        alpha=alpha,
        theta=theta,
        phi=phi,
        dtype=dtype,
        spacing=spacing,
    )

    trainable0, static0 = partition_with_freeze(model0, freeze=freeze)
    loss_fn = loss(static0, data)

    trainable_opt, losses, grad_norms = fit_lbfgs(
        loss_fn, trainable0, max_iter=max_iter, tol=tol
    )
    model_opt = eqx.combine(trainable_opt, static0)

    positions = model_opt()
    parameters = model_opt.parameter_dict()
    positions.block_until_ready()
    positions_np = np.asarray(positions)

    return {
        "losses": np.array(losses),
        "grad_norms": np.array(grad_norms),
        "model_points": positions_np,
        "parameters": parameters,
    }
