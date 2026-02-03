from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .loss import loss
from .models.default_params import default_params
from .models.models import apply_init
from .models.registry import get_model_cls
from .optim import fit_lbfgs
from .utils import Data, partition_with_freeze


def normalize_dtype(dtype):
    if dtype in ("float32", "fp32"):
        return jnp.float32
    if dtype in ("float64", "fp64"):
        return jnp.float64
    return dtype


def run_locmofit(
    model_name: str,
    locs: np.ndarray | jnp.ndarray,
    stddev: np.ndarray | jnp.ndarray,
    *,
    init_params: Optional[Mapping[str, Any]] = None,
    freeze: tuple[str, ...] = (),
    max_iter: int = 200,
    spacing: float = 3.0,  # 3 nm
    dtype: str = "float32",
    tol: float = 1e-6,
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

    dtype = normalize_dtype(dtype)

    locs_j = jnp.asarray(locs, dtype=dtype)
    stddev_j = jnp.asarray(stddev, dtype=dtype)
    data = Data.from_arrays(locs_j, stddev_j)

    ModelCls = get_model_cls(model_name)
    dparams = default_params(ModelCls)
    updated_params = apply_init(dparams, dict(init_params or {}))
    model0 = ModelCls.init(
        params=updated_params,
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
