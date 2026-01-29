from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array


def fit_lbfgs(
    loss_fn: Callable[[Any], Any],
    params0: Any,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[Any, Array, Array]:
    """
    Generic Optax L-BFGS solver.

    Args:
      loss_fn: function(params) -> scalar loss (JAX scalar)
      params0: initial params pytree
      max_iter: maximum iterations
      tol: stop when grad_norm < tol (after at least 1 iter)

    Returns:
      params_opt, info dict with keys: final_loss, iters, grad_norm
    """
    optimizer = optax.lbfgs()
    opt_state0 = optimizer.init(params0)
    value_and_grad_fun = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def _solve(params, opt_state):
        def step(carry):
            params, state, losses, grad_norms = carry

            value, grad = value_and_grad_fun(params, state=state)
            gn = optax.tree.norm(grad)

            k = optax.tree.get(state, "count")  # take index BEFORE update increments it
            losses = losses.at[k].set(value)
            grad_norms = grad_norms.at[k].set(gn)

            updates, state = optimizer.update(
                grad, state, params, value=value, grad=grad, value_fn=loss_fn
            )
            params = optax.apply_updates(params, updates)

            return (params, state, losses, grad_norms)

        def cond(carry):
            _, state, _, _ = carry
            k = optax.tree.get(state, "count")
            g = optax.tree.get(state, "grad")
            return (k == 0) | ((k < max_iter) & (optax.tree.norm(g) >= tol))

        losses = jnp.full((max_iter,), jnp.nan, dtype=params0.dtype)
        grad_norms = jnp.full((max_iter,), jnp.nan, dtype=params0.dtype)

        init = (
            params,
            opt_state,
            losses,
            grad_norms,
        )
        return jax.lax.while_loop(cond, step, init)

    params_opt, state_opt, losses, grad_norms = _solve(params0, opt_state0)

    k_final = optax.tree.get(state_opt, "count")
    losses = losses[:k_final]
    grad_norms = grad_norms[:k_final]
    return params_opt, losses, grad_norms
