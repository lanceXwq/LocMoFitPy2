from typing import Any, Callable, Dict, Tuple

import jax
import optax


def fit_lbfgs(
    loss_fn: Callable[[Any], Any],
    params0: Any,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[Any, Dict[str, float]]:
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
            params, state = carry
            value, grad = value_and_grad_fun(params, state=state)
            updates, state = optimizer.update(
                grad, state, params, value=value, grad=grad, value_fn=loss_fn
            )
            params = optax.apply_updates(params, updates)
            return params, state

        def cond(carry):
            _, state = carry
            k = optax.tree.get(state, "count")
            g = optax.tree.get(state, "grad")
            return (k == 0) | ((k < max_iter) & (optax.tree.norm(g) >= tol))

        return jax.lax.while_loop(cond, step, (params, opt_state))

    params_opt, state_opt = _solve(params0, opt_state0)

    # Diagnostics (host-friendly scalars)
    final_loss = loss_fn(params_opt)
    final_loss.block_until_ready()

    iters = int(optax.tree.get(state_opt, "count"))
    grad = optax.tree.get(state_opt, "grad")
    grad_norm = optax.tree.norm(grad)
    grad_norm.block_until_ready()

    info = {
        "final_loss": float(final_loss),
        "iters": iters,  # keep numeric; MATLAB will cast anyway
        "grad_norm": float(grad_norm),
    }
    return params_opt, info
