from typing import Any, Dict

import numpy as np

import locmofitpy2


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


def run_locmofit(model_name: str, locs_np: np.ndarray, stddev_np: np.ndarray, **kwargs):
    return locmofitpy2.run_locmofit(model_name, locs_np, stddev_np, **kwargs)
