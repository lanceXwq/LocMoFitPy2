from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import locmofitpy2


def _get_model_cls(model_name: str):
    # Allows "SphericalCap" -> locmofitpy2.SphericalCap
    try:
        return getattr(locmofitpy2, model_name)
    except AttributeError as e:
        raise ValueError(
            f"Unknown model_name={model_name!r}. "
            f"Expected a class in locmofitpy2 (e.g. 'SphericalCap')."
        ) from e


def _params_to_plain_dict(model: Any) -> Dict[str, float]:
    """
    Convert model fields into a MATLAB-friendly dict.
    This tries a few common patterns:
      - model.<name> is a scalar JAX array
      - model has a __dict__ of fields
    You can customize per-model if needed.
    """
    out: Dict[str, float] = {}

    # Prefer explicit known names if present; extend as you like.
    preferred = ("x", "y", "z", "r", "c", "vartheta", "phi0", "theta", "phi", "delta_z")
    for k in preferred:
        if hasattr(model, k):
            v = getattr(model, k)
            try:
                out[k] = float(v)
            except TypeError:
                pass

    # Fallback: scan __dict__ and convert scalar arrays
    if hasattr(model, "__dict__"):
        for k, v in model.__dict__.items():
            if k.startswith("_") or k in out:
                continue
            try:
                # Convert 0-d arrays / scalar-like
                out[k] = float(v)
            except Exception:
                continue

    return out


def run_locmofit(
    model_name: str,
    locs_np: np.ndarray,  # (N,3)
    stddev_np: np.ndarray,  # (N,K)
    *,
    seed: int = 1,
    model_init_kwargs: Dict[str, Any] | None = None,
    freeze: Tuple[str, ...] = (),
    max_iter: int = 200,
    tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    MATLAB-facing entry point.

    Returns:
      {
        "final_loss": float,
        "params": {str: float},
        "positions": np.ndarray
      }
    """
    if model_init_kwargs is None:
        model_init_kwargs = {}

    locs = jnp.asarray(locs_np, dtype=jnp.float32)
    stddev = jnp.asarray(stddev_np, dtype=jnp.float32)
    data = locmofitpy2.Data.from_arrays(locs, stddev)

    key = jax.random.PRNGKey(seed)
    ModelCls = _get_model_cls(model_name)

    # Convention: your classes have .init(key, **kwargs)
    model0 = ModelCls.init(key, **model_init_kwargs)

    trainable0, static0 = locmofitpy2.partition_with_freeze(model0, freeze=freeze)
    loss = locmofitpy2.loss(static0, data)

    optimizer = optax.lbfgs()
    opt_state0 = optimizer.init(trainable0)
    value_and_grad_fun = optax.value_and_grad_from_state(loss)

    @jax.jit
    def solve_lbfgs(trainable, opt_state):
        def step(carry):
            trainable, state = carry
            value, grad = value_and_grad_fun(trainable, state=state)
            updates, state = optimizer.update(
                grad, state, trainable, value=value, grad=grad, value_fn=loss
            )
            trainable = optax.apply_updates(trainable, updates)
            return trainable, state

        def cond(carry):
            _, state = carry
            k = optax.tree.get(state, "count")
            g = optax.tree.get(state, "grad")
            return (k == 0) | ((k < max_iter) & (optax.tree.norm(g) >= tol))

        return jax.lax.while_loop(cond, step, (trainable, opt_state))

    trainable_opt, _ = solve_lbfgs(trainable0, opt_state0)

    model_opt = locmofitpy2.combine(trainable_opt, static0)

    final_loss = loss(trainable_opt)
    final_loss.block_until_ready()

    # Convention: calling model returns positions
    positions = model_opt()
    positions.block_until_ready()
    positions_np = np.asarray(positions)

    return {
        "final_loss": float(final_loss),
        "params": _params_to_plain_dict(model_opt),
        "positions": positions_np,
    }
