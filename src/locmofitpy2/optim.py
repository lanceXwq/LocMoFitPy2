from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array


def tree_l2_norm(t):
    return jnp.sqrt(
        sum([jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(t)])
    )


def tree_dot(a, b):
    return sum(
        [
            jnp.vdot(x, y)
            for x, y in zip(
                jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
            )
        ]
    )


def log_state(step, value, grad, updates, opt_state):
    ls_state = opt_state[2]
    info = ls_state.info

    gnorm = optax.global_norm(grad)
    unorm = tree_l2_norm(updates)
    cos = tree_dot(grad, updates) / (gnorm * unorm + 1e-30)

    jax.debug.print(
        "k={k} loss={loss:.6e} |g|={gn:.3e} |Δ|={un:.3e} "
        "g·Δ/|g||Δ|={cos:.3f} lr={lr:.3e} ls={nls}",
        k=step,
        loss=value,
        gn=gnorm,
        un=unorm,
        cos=cos,
        lr=ls_state.learning_rate,
        nls=info.num_linesearch_steps,
    )


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
    optimizer = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=100,
            verbose=False,
            initial_guess_strategy="one",
        )
    )

    opt_state0 = optimizer.init(params0)
    value_and_grad_fun = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def _solve(params, opt_state):
        def step(carry):
            params, state, losses, grad_norms = carry

            value, grad = value_and_grad_fun(params, state=state)
            gn = optax.tree.norm(grad)

            # take index BEFORE update increments it
            k = optax.tree.get(state, "count")
            losses = losses.at[k].set(value)
            grad_norms = grad_norms.at[k].set(gn)

            updates, state = optimizer.update(
                grad, state, params, value=value, grad=grad, value_fn=loss_fn
            )

            # log_state(
            #     k,
            #     value,
            #     grad,
            #     updates,
            #     state,
            # )

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

    k_final = int(jax.device_get(optax.tree.get(state_opt, "count")))
    losses = losses[:k_final]
    grad_norms = grad_norms[:k_final]
    return params_opt, losses, grad_norms
