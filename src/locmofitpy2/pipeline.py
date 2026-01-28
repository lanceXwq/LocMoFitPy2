from __future__ import annotations

# from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .loss import loss
from .optim import fit_lbfgs
from .registry import get_model_cls
from .utils import Data, partition_with_freeze


def run_locmofit(
    model_name: str,
    locs: np.ndarray | jnp.ndarray,
    stddev: np.ndarray | jnp.ndarray,
    *,
    seed: int = 1,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    freeze: Tuple[str, ...] = (),
    max_iter: int = 200,
    tol: float = 1e-6,
    dtype=jnp.float32,
) -> Dict[str, Any]:
    """
    High-level fitting pipeline.

    Returns a RunResult containing:
      - fitted model object
      - optimized trainable/static parts
      - positions as NumPy array on host
      - LBFGS diagnostics (final loss, iters, grad_norm)
    """
    if model_init_kwargs is None:
        model_init_kwargs = {}

    locs_j = jnp.asarray(locs, dtype=dtype)
    stddev_j = jnp.asarray(stddev, dtype=dtype)
    data = Data.from_arrays(locs_j, stddev_j)

    key = jax.random.PRNGKey(seed)
    ModelCls = get_model_cls(model_name)
    model0 = ModelCls.init(key, **model_init_kwargs)

    trainable0, static0 = partition_with_freeze(model0, freeze=freeze)
    loss_fn = loss(static0, data)

    trainable_opt, info = fit_lbfgs(loss_fn, trainable0, max_iter=max_iter, tol=tol)
    model_opt = eqx.combine(trainable_opt, static0)

    positions = model_opt()
    parameters = model_opt.parameter_dict()
    positions.block_until_ready()
    positions_np = np.asarray(positions)

    return {
        "final_loss": float(info["final_loss"]),
        "iters": float(info["iters"]),
        "grad_norm": float(info["grad_norm"]),
        "model_points": positions_np,
        "parameters": parameters,
    }
