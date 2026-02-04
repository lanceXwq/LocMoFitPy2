from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .loss import make_loss
from .models.default_params import default_params
from .models.registry import get_model_cls
from .models.utils import apply_init
from .optim import fit_lbfgs
from .utils import Data, partition_with_freeze


def _normalize_freeze(freeze: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(freeze, str):
        return (freeze,)
    return tuple(freeze)


def run_locmofit(
    model_name: str,
    locs: ArrayLike,
    loc_precisions: ArrayLike,
    *,
    init_params: Mapping[str, Any] | None = None,
    freeze: str | Sequence[str] = (),
    max_iter: int = 200,
    spacing: float = 3.0,  # 3 nm
    dtype: Any = "float32",
    tol: float = 1e-6,
) -> dict[str, Any]:
    dtype = jnp.dtype(dtype)

    locs_j = jnp.asarray(locs, dtype=dtype)
    prec_j = jnp.asarray(loc_precisions, dtype=dtype)
    if locs_j.ndim != 2 or locs_j.shape[1] != 3:
        raise ValueError(f"`locs` must be (N,3), got {locs_j.shape}")

    if prec_j.shape != locs_j.shape:
        raise ValueError(
            f"`stddev/precisions` must match `locs` shape; got {prec_j.shape} vs {locs_j.shape}"
        )
    data = Data.from_arrays(locs_j, prec_j)

    ModelCls = get_model_cls(model_name)
    dparams = default_params(ModelCls)
    updated_params = apply_init(dparams, dict(init_params or {}))
    model0 = ModelCls.init(
        params=updated_params,
        dtype=dtype,
        spacing=spacing,
    )

    freeze = _normalize_freeze(freeze)
    trainable0, static0 = partition_with_freeze(model0, freeze=freeze)
    loss_fn = make_loss(static0, data)

    trainable_opt, losses, grad_norms = fit_lbfgs(
        loss_fn, trainable0, max_iter=max_iter, tol=tol
    )
    model_opt = eqx.combine(trainable_opt, static0)

    positions = model_opt()

    return {
        "losses": np.asarray(jax.device_get(losses)),
        "grad_norms": np.asarray(jax.device_get(grad_norms)),
        "model_points": np.asarray(jax.device_get(positions)),
        "parameters": model_opt.parameter_dict(),
    }
