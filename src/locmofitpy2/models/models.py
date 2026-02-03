import warnings
from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp

from .registry import register_model
from .spcap import SphericalCap

register_model("SphericalCap", SphericalCap, aliases=("spcap", "sphericalcap"))


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


def apply_init(default: Mapping[str, Any], init: Mapping[str, Any]) -> dict[str, Any]:
    allowed = set(default)
    unknown = [k for k in init if k not in allowed]
    if unknown:
        warnings.warn(
            f"Ignoring unknown init keys {unknown!r}. Allowed keys: {sorted(allowed)!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    out = dict(default)
    out.update({k: v for k, v in init.items() if k in allowed})
    return out
