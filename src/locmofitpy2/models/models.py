from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp


def _get_attr_path(obj: Any, path: str) -> Any:
    """Get a (possibly nested) attribute via dot-separated path, e.g. 'foo.bar.c'."""
    cur = obj
    for name in path.split("."):
        cur = getattr(cur, name)
    return cur


def set_params(
    model: Any,
    overrides: Mapping[str, Any],
) -> Any:
    """
    Functionally override fields in an Equinox/JAX pytree model.

    Args:
      model: eqx.Module (or pytree containing eqx.Modules)
      overrides: mapping from field-name or dot-path -> value

    Returns:
      New model with overrides applied.
    """
    out = model

    for path, value in overrides.items():
        repl = jnp.asarray(value)

        out = eqx.tree_at(
            lambda t, p=path: _get_attr_path(t, p),
            out,
            repl,
        )

    return out
