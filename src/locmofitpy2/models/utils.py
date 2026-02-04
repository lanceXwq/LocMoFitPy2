import warnings
from typing import Any, Mapping

import numpy as np


def to_py_or_np(x: Any):
    a = np.asarray(x)
    if a.size == 1:
        return a.reshape(()).item()
    return a


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
